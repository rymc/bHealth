"""
This is an example dealing with SPHERE challenge data using LSTM.
The data can be downloaded using below link: 
https://data.bris.ac.uk/data/dataset/8gccwpx47rav19vk8x4xapcog
The downloaded 'metadata' should be placed in folder: '../SCHLNG_data/metadata', 
and the 'train' and 'test' in folder '../SCHLNG_data'.
"""

from __future__ import print_function

# For number crunching
import numpy as np
import pandas as pd

# for date and time
import datetime

# For visualisation
import matplotlib.pyplot as pl 
import seaborn as sns 

# For prediction 
import sklearn

# Misc
from itertools import cycle
import json 
import os
import sys
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn import  metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.feature_selection import SelectFromModel

sys.path.append('../')
from bhealth.visualisations import plot_metrics
from bhealth.visualisations import features_figure
from bhealth.visualisations import plot_features
from bhealth.visualisations import plot_test_train_splits
from bhealth import data_loading
from bhealth.transforms import Transforms
from bhealth.metrics import Metrics
from bhealth.metric_wrappers import Wrapper

import scipy.stats
from scipy.signal import butter, filtfilt
from visualise_data_LSTM import Sequence
from visualise_data_LSTM import SequenceVisualisation

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable 
import math

# Recurrent neural network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden units and memory units
        h0 = torch.rand(self.num_layers, x.shape[0], self.hidden_size).to(device) #x.size(0)是batch_size
        c0 = torch.rand(self.num_layers, x.shape[0], self.hidden_size).to(device)
    
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

SCHLNG_data_path = '../SCHLNG_data'
metadata_path = '../SCHLNG_data/metadata'    

def load_sequence(file_id):
    filename = str(file_id).zfill(5)

    df = pd.read_csv('{}/train/{}/columns_200ms.csv'.format(SCHLNG_data_path, filename))
    data = df.values
    target = np.asarray(pd.read_csv('{}/train/{}/targets.csv'.format(SCHLNG_data_path, filename)))[:, 2:]

    return data, target


def load_sequences(file_ids):
    x_es = []
    y_es = []

    for file_id in file_ids:
        data, target = load_sequence(file_id)

        x_es.append(data)
        y_es.append(target)

    return np.row_stack(x_es), np.row_stack(y_es)


def get_raw_ts_X_y():
    sns.set_context('poster')
    sns.set_style('darkgrid')
    current_palette = cycle(sns.color_palette())
    
    ###### Add the parent folder to PYTHONPATH (sys.path)
    nb_dir = os.path.split(os.getcwd())[0]
    if nb_dir not in sys.path:
        sys.path.append(nb_dir)        

    warnings.filterwarnings('ignore')
    
    """
    For every data modality, we will extract some very simple features: the mean, min, max, median, and standard 
    deviation of one second windows. We will put function pointers in a list called 'feature_functions' so that we 
    can easily call these on our data for later
    """
    
    feature_functions = [np.mean]
    feature_names = ['mean']
    
    # We will keep the number of extracted feature functions as a parameter 
    num_ff = len(feature_functions)
    
    # We will want to keep track of the feature names for later, so we will collect these in the following list: 
    column_names = []
    
    # These are the modalities that are available in the dataset, and the .iterate() function returns the data 
    # in this order
    modality_names = ['acceleration', 'rssi', 'pir', 'video_living_room', 'video_kitchen', 'video_hallway']
    
    """
    Iterate over all training directories
    """
    for train_test in ('train', 'test', ): 
        if train_test is 'train': 
            print ('Extracting features from training data.\n')
        else: 
            print ('\n\n\nExtracting features from testing data.\n')
            
        for fi, file_id in enumerate(os.listdir('{}/{}/'.format(SCHLNG_data_path, train_test))):
            stub_name = str(file_id).zfill(5)
    
            if train_test is 'train' or np.mod(fi, 50) == 0:
                print ("Starting feature extraction for {}/{}".format(train_test, stub_name))
    
            # Use the sequence loader to load the data from the directory. 
            data = Sequence(metadata_path, '{}/{}/{}'.format(SCHLNG_data_path, train_test, stub_name))
            data.load()
    
            """
            Populate the column_name list here. This needs to only be done on the first iteration
            because the column names will be the same between datasets. 
            """
            if len(column_names) == 0:
                for lu, modalities in data.iterate():
                    for modality, modality_name in zip(modalities, modality_names):
                        for column_name, column_data in modality.transpose().iterrows():
                            for feature_name in feature_names:
                                column_names.append('{0}_{1}_{2}'.format(modality_name, column_name, feature_name))
    
                    # Break here 
                    break 
    
            rows = []
    
            for ri, (lu, modalities) in enumerate(data.iterate()):
                row = [] 
                for modality in modalities:
                    """
                    The accelerometer dataframe, for example, has three columns: x, y, and z. We want to extract features 
                    from all of these, and so we iterate over the columns here. 
                    """
                    for name, column_data in modality.transpose().iterrows():
                        if len(column_data) > 3:
                            """
                            Extract the features stored in feature_functions on the column data if there is sufficient 
                            data in the dataframe. 
                            """
                            row.extend(map(lambda ff: ff(column_data), feature_functions))
    
                        else:
                            """
                            If no data is available, put nan placeholders to keep the column widths consistent
                            """
                            row.extend([np.nan] * num_ff)
    
                # Do a quick sanity check to ensure that the feature names and number of extracted features match
                assert len(row) == len(column_names)
    
                # Append the row to the full set of features
                rows.append(row)
    
                # Report progress 
                if train_test is 'train':
                    if np.mod(ri + 1, 50) == 0:
                        print ("{:5}".format(str(ri + 1))),
    
                    if np.mod(ri + 1, 500) == 0:
                        print
    
            """
            We will save these features to a new file called 'columns.csv' for use later. This file will be located 
            in the name of the training sequence. 
            """
            df = pd.DataFrame(rows)
            df.columns = column_names
            df.to_csv('{}/{}/{}/columns_200ms.csv'.format(SCHLNG_data_path, train_test, stub_name), index=False)
    
            if train_test is 'train' or np.mod(fi, 50) == 0:
                if train_test is 'train': print 
                print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))
                
    column_names            
    
    # Load the training and testing data     
    train_x, train_y = load_sequences([1])
    tmpy = train_y
    #Sub-sampling the labels'
    train_y = np.zeros((tmpy.shape[0]*5, tmpy.shape[1]),np.float64) 
    for i in range(0, tmpy.shape[0]):
        for j in range(0, tmpy.shape[1]):
            train_y[i*5:i*5+5,j] = tmpy[i,j]   
    
    # We will want to impute the missing data 
    imputer = Imputer()
    imputer.fit(train_x)
    train_x = imputer.transform(train_x)
    
    # Load the label names 
    labels = json.load(open(metadata_path + '/annotations.json'))
    n_classes = len(labels)
    
    train_y_has_annotation = np.isfinite(train_y.sum(1))
    train_x = train_x[train_y_has_annotation]
    train_y = train_y[train_y_has_annotation]
    train_y = np.argmax(train_y, axis=1)
        
    tslist = []
    tsStart = 1491735899;       
    for idx in range(0, train_x.shape[0]-1):
        times = tsStart + idx
        local = datetime.datetime.fromtimestamp(times)        
        tslist_ = local.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        tslist.append(tslist_)
    ts = tslist       
    
    return ts, train_x, train_y

def preprocess_X_y_FeaExtr(ts, X, y):
    new_X = []
    new_y = []

    winlength_seconds = 1
    overlap_seconds = 0
    print("Window size of "+str(winlength_seconds)+" seconds and overlap of"+str(float(overlap_seconds)*100/ winlength_seconds)+"%")
    samples_per_sec = 5
    winlength = samples_per_sec * winlength_seconds
    current = winlength
    overlap = samples_per_sec * overlap_seconds

    transform = Transforms(window_length=winlength, window_overlap=overlap)
    print("Use number of mean crossings, spectral entropy as features...")
    feature_transforms = [transform.mean,
                          transform.std,
                          transform.prod,
                          transform.minx,
                          transform.maxx]
#                          transform.sumx]
#                          transform.mean_crossings]
#                          transform.spec_entropy,
#                          transform.zero_crossings]
#                          transform.interq,
#                          transform.skewn,
#                          transform.spec_energy,
#                          transform.p25,
#                          transform.p75,
#                          transform.kurtosis]

    while True:
        windowed_raw = transform.slide(X)
        if len(windowed_raw) > 0:
            try:
                # TODO Save the timestamps into a windowed_ts instead
                windowed_features = [ts[transform.current_position][0]]
            except Exception as e:
                print(e)
                break
            for function in feature_transforms:
                windowed_features.extend((np.apply_along_axis(function, 0, windowed_raw).tolist()))
            new_X.append(windowed_features)

            windowed_raw_labels = transform.slide(y, update=False)
            most_freq_label = np.bincount(windowed_raw_labels).argmax()
            new_y.append(most_freq_label)

    # Convert lists to Numpy arrays
    new_X = np.array(new_X)
    # Store new timestamps
    # TODO This will not be necessary after the windowed_ts is generated
    ts = new_X[:, 0]
    # TODO Check why the features are converted to strings
    # Remove datetime from features
    new_X = new_X[:, 1:].astype(float)
    new_y = np.array(new_y)

    return ts, new_X, new_y

def preprocess_X_y_Shape(ts, X, y):
    new_X = []
    new_y = []

    winlength_seconds = 10
    overlap_seconds = 9
    print("Window size of "+str(winlength_seconds)+" seconds and overlap of"+str(float(overlap_seconds)*100/ winlength_seconds)+"%")
    samples_per_sec = 1
    winlength = samples_per_sec * winlength_seconds
    current = winlength
    overlap = samples_per_sec * overlap_seconds

    transform = Transforms(window_length=winlength, window_overlap=overlap)
    print("Use number of mean crossings, spectral entropy as features...")
    feature_transforms = [transform.raw]

    while True:
        windowed_raw = transform.slide(X)
        if len(windowed_raw) > 0:
            try:
                windowed_features = [ts[transform.current_position][0]]
            except Exception as e:
                print(e)
                break
            for function in feature_transforms:
                windowed_features.extend((np.apply_along_axis(function, 0, windowed_raw).tolist()))
            new_X.append(windowed_features)

            windowed_raw_labels = transform.slide(y, update=False)
            most_freq_label = np.bincount(windowed_raw_labels).argmax()
            new_y.append(most_freq_label)

    # Convert lists to Numpy arrays
    new_X = np.array(new_X)
    # Store new timestamps
    ts = new_X[:, 0]
    # Remove datetime from features
    new_X = new_X[:, 1:]
    new_y = np.array(new_y)

    dims = np.size(new_X[0][0])    
    tmpX = np.ones((new_X.shape[0],new_X.shape[1],dims), dtype='float64')
    for i in range(0, new_X.shape[0]):
        for j in range(0, new_X.shape[1]):
            for k in range(0, dims):
                tmpX[i][j][k]=new_X[i][j][k]
                
    new_X = tmpX
    return ts, new_X, new_y


def split_train_test(X, y):
    # Create train and test partitions
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    train_index, test_index = skf.split(X, y).__next__()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return (X_train, y_train), (X_test, y_test)


def get_classifier_grid():
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    models = {'rf': {'model': RandomForestClassifier(),
                     'parameters': {'n_estimators': [200, 250],
                                    'min_samples_leaf': [1, 5, 10]}},
              'lr': {'model': LogisticRegression(penalty='l2'),
                     'parameters': {'C': [0.01, 0.1, 1, 10, 100]}},
              'svc': {'model': SVC(probability=True),
                      'parameters': {'kernel': ['linear'],
                                      'C': [1, 10, 100, 1000]}},
              'svc-rbf': {'model': SVC(probability=True),
                          'parameters': {'kernel': ['rbf'],
                                          'gamma': [1e-3, 1e-4],
                                          'C': [1, 10, 100, 1000]}},
              }

    classifier_name = 'lr'

    steps = [('imputer', SimpleImputer(missing_values=np.nan,
                                       strategy='mean')),
             ('scaler', StandardScaler()),
             ('clf', models[classifier_name]['model'])]

    pipeline = Pipeline(steps)

    pipeline_parameters = {'clf__' + key: value for key, value in
                           models[classifier_name]['parameters'].items()}

    clf_grid = GridSearchCV(pipeline, param_grid=pipeline_parameters, cv=cv,
                            refit=True)
    return clf_grid


def print_summary(clf_grid, X_test, y_test):
    import numpy as np
    print('Best parameters are: {}'.format(clf_grid.best_params_))
    print("CV accuracy "+str(np.mean(clf_grid.cv_results_['mean_test_score'])))

    # The best model was fitted on the full training data, here it is tested only
    tt_score = clf_grid.score(X_test, y_test)
    print("Train / test split accuracy "+str(tt_score))


def brier_score(given, predicted, weight_vector): 
    return np.power(given - predicted, 2.0).dot(weight_vector).mean()

if __name__ == '__main__':
    tsraw, Xraw, yraw = get_raw_ts_X_y()    
    tsfea, Xfea, yfea = preprocess_X_y_FeaExtr(tsraw, Xraw, yraw)
    ts, X, y = preprocess_X_y_Shape(tsfea, Xfea, yfea)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)

    #Layer normalization for LSTM train
#    eps = 1e-5
#    Xln_train = X_train    
#    for j in range(Xln_train.shape[0]):
#        for k in range(Xln_train.shape[1]):
#            x_mean = np.mean(Xln_train[j,k,:])
#            x_var = np.var(Xln_train[j,k,:])
#            Xln_train[j,k,:] = (Xln_train[j,k,:] - x_mean) / np.sqrt(x_var + eps)  
#        
#    #Layer normalization for LSTM test
#    Xln_test = X_test    
#    for j in range(Xln_test.shape[0]):
#        for k in range(Xln_test.shape[1]):
#            x_mean = np.mean(Xln_test[j,k,:])
#            x_var = np.var(Xln_test[j,k,:])
#            Xln_test[j,k,:] = (Xln_test[j,k,:] - x_mean) / np.sqrt(x_var + eps)           
  
    device = torch.device('cpu')
    # Hyper-parameters
    input_size = X_train.shape[2]       # input feature dimension
    hidden_size = 600      # hidden size
    num_layers = 1         # LSTM layer
    num_classes = 20       # num of classes
    batch_size = 50       # batch size for optimization
    num_epochs = 10        # num of epochs
    learning_rate = 0.01
    # learning rate    
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)    
    
    # Loss and optimizer
#    ClassWeights=np.array([1.352985,1.386846,1.595874,1.353187,0.347784,0.661082,1.047236,0.398865,
#                  0.207586,1.505783,0.110181,1.078033,1.365604,1.170241,1.193364,1.18037,
#                  1.344149,1.116838,1.080839,0.503152])
#    ClassWeights = torch.from_numpy(ClassWeights).float()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    if len(X_train) % batch_size == 0:
        train_step = len(X_train)//batch_size        
    else:
        train_step = len(X_train)//batch_size + 1  
    
    Loss = np.zeros((num_epochs, train_step))
    Accr = np.zeros((num_epochs, train_step))
    LossTest = np.zeros((num_epochs, train_step))
    AccrTest = np.zeros((num_epochs, train_step))
    for epoch in range(num_epochs):        
        print('Epoch',epoch)
        randIdx = torch.randperm(len(X_train))   # Random Sampler
        for i in range(train_step):
            print('Iteration',i)
            if i < train_step:
                idxBatch = randIdx[i*batch_size:(i+1)*batch_size]
            else:
                idxBatch = randIdx[i*batch_size:len(X_train)]       
            
            trainXbatch = X_train[idxBatch,:,:]   
            trainXbatch = torch.from_numpy(trainXbatch).float() #Convert ndarray to Tensor        
            trainy = y_train[idxBatch]
            if trainy.size > 1:   #torch.from_numpy only applies to array with more than one item.
                trainy = torch.from_numpy(trainy)
            else:              #the last batch
                trainXbatch = trainXbatch.reshape(-1,trainXbatch.shape[0],trainXbatch.shape[1])
                trainy = np.array([trainy])
                trainy = torch.from_numpy(trainy)
            print('size',trainXbatch.shape)
            
            # Forward pass            
            outputs = model(trainXbatch)                
            print(outputs.size())
            loss = criterion(outputs, trainy)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            
    # Test the mode on training set
    with torch.no_grad():
        correct = 0
        total = 0
        for j in range(len(X_train)):
            trainX = torch.from_numpy(X_train[j,:,:])
            trainy = y_train[j]
            trainX = trainX.reshape(-1,trainX.shape[0],trainX.shape[1]).float()
            outputs = model(trainX)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == trainy).sum().item()            
        print('Train Accuracy: {} %'.format(100 * correct / total)) 

    # Test the model on testing set
    with torch.no_grad():
        correct = 0
        total = 0
        for j in range(len(X_test)):
            testX = torch.from_numpy(X_test[j,:,:])
            testy = y_test[j]
            testX = testX.reshape(-1,testX.shape[0],testX.shape[1]).float()
            outputs = model(testX)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == testy).sum().item()
            
    AccrTest[epoch, i] = correct / total        
    print('Test Accuracy: {} %'.format(100 * correct / total))   

    '''Shallow learning methods'''
    X_shallow_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
    y_shallow_train = y_train
    X_shallow_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])
    y_shallow_test = y_test
    
    clf_grid = get_classifier_grid()
    clf_grid.fit(X_shallow_train, y_shallow_train)
    y_prob = clf_grid.predict_proba(X_shallow_test)
    y_pred = np.argmax(y_prob, axis=1)    
    '''
    cfm = confusion_matrix(y_test, y_pred)    
    class_weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    y_test_prob = np.zeros((y_test.shape[0], max(y_train)+1),np.float64) 
    for idx in range(0, y_test.shape[0]):
        y_test_prob[idx,y_test[idx]] = 1
    bs = brier_score(y_test_prob, y_prob, class_weights)'''
    print_summary(clf_grid, X_shallow_test, y_shallow_test)  