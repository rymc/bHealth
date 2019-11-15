import os
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bhealth.visualisations import plot_metrics
from bhealth.visualisations import features_figure
from bhealth.visualisations import plot_features
from bhealth.visualisations import plot_test_train_splits
from bhealth import data_loading
from bhealth import data_loading_debug

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from bhealth.transforms import Transforms
from bhealth.metrics import Metrics
from bhealth.metric_wrappers import Wrapper

import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable 

import math
from PIL import Image
from pylab import *

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

def get_raw_ts_X_y():

    labels, ts, xyz = data_loading_debug.data_loader_accelerometer_debug()
    return ts, xyz, labels

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
                          transform.mean_crossings,
                          transform.spec_entropy,
                          transform.zero_crossings,
                          transform.interq,
                          transform.skewn,
                          transform.spec_energy,
                          transform.p25,
                          transform.p75,
                          transform.kurtosis]

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
    plot_test_train_splits(y_train, y_test)
    return (X_train, y_train), (X_test, y_test)


def get_classifier_grid():
    # Create cross-validation partitions from training
    # This should select the best set of parameters
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    clf = RandomForestClassifier()
    param_grid = {'n_estimators' : [200, 250],
                  'min_samples_leaf': [5, 10]}
    clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True,
                            iid=True)
    return clf_grid


def print_summary(clf_grid, X_test, y_test):
    import numpy as np
    print('Best parameters are: {}'.format(clf_grid.best_params_))
    print("CV accuracy "+str(np.mean(clf_grid.cv_results_['mean_test_score'])))

    # The best model was fitted on the full training data, here it is tested only
    tt_score = clf_grid.score(X_test, y_test)
    print("Train / test split accuracy "+str(tt_score))


def activity_metrics(labels, timestamps, span):
    """Outputs typical activity metrics."""

    descriptor_map = {
                'sitting' : [1],
                'walking' : [2],
                'washing' : [3],
                'eating'  : [4],
                'sleeping': [5],
                'studying': [6]
            }

    if not os.path.exists('./output/'):
        os.mkdir('./output/')

    metrics = Wrapper(labels, timestamps, span, 1, 25, descriptor_map,
                      adjecency=None)

    df_time = timestamps.astype('datetime64')
    df_time = pd.DataFrame(df_time, columns=['Time'])
    df_label = pd.DataFrame(labels, columns=['Label'])

    metric_array= [metrics.duration_sitting,
                   metrics.duration_walking,
                   metrics.duration_washing,
                   metrics.duration_eating,
                   metrics.duration_sleeping,
                   metrics.duration_studying,
                   metrics.number_of_unique_activities]

    metric_container, date_container = metrics.run_metric_array(
        metric_array, csv='./output/accelerometer_metrics.csv')

    return metric_container, date_container

if __name__ == '__main__':
    ts, X, y = get_raw_ts_X_y()
    ts, X, y = preprocess_X_y_FeaExtr(ts, X, y)    
    ts, X, y = preprocess_X_y_Shape(ts, X, y)    
    
    #modify [77-82] to [1-6] to suit LSTM
    for idx in range(y.shape[0]):
        if y[idx]>0:
            y[idx] -= 76
   
    #split the train-test 
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)    
 
    # Device configuration
    device = torch.device('cpu')
    # Hyper-parameters
    input_size = X_train.shape[2]        # input feature dimension
    hidden_size = 50      # hidden size
    num_layers = 1        # LSTM layer
    num_classes = 7       # num of classes
    batch_size = 50       # batch size for optimization
    num_epochs = 5        # num of epochs
    learning_rate = 0.03  # learning rate    
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)    
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)      
                                      
    # Train the model
    if len(X_train) % batch_size == 0:
        train_step = len(X_train)//batch_size        
    else:
        train_step = len(X_train)//batch_size + 1  

    Loss = np.zeros(train_step)
    for epoch in range(num_epochs):
        randIdx = torch.randperm(len(X_train))
        for i in range(train_step):
            if i < train_step:
                idxBatch = randIdx[i*batch_size:(i+1)*batch_size]
            else:
                idxBatch = randIdx[i*batch_size:len(X_train)]
                    
            idxBatch = randIdx[i*batch_size:(i+1)*batch_size]
            trainXbatch = X_train[idxBatch,:,:]   
#            trainXbatch=trainXbatch.reshape(-1,trainXbatch.shape[0],trainXbatch.shape[1])
            trainXbatch = torch.from_numpy(trainXbatch).float()  
            trainy = y_train[idxBatch]   
            trainy = torch.from_numpy(trainy)
            #trainy = torch.from_numpy(np.array(trainy))

            # Forward pass
            outputs = model(trainXbatch)
            loss = criterion(outputs, trainy)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, train_step, loss.item()))
            Loss[i]=loss.item()

    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(len(X_train)):
            trainX = torch.from_numpy(X_train[i,:,:])
            trainy = y_train[i]
            trainX = trainX.reshape(-1,trainX.shape[0],trainX.shape[1]).float()
            outputs = model(trainX)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == trainy).sum().item()
    
        print('Train Accuracy: {} %'.format(100 * correct / total))  
        
    # Test the model    
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(len(X_test)):
            testX = torch.from_numpy(X_test[i,:,:])
            testy = y_test[i]
            outputs = model(testX.reshape(-1,testX.shape[0],testX.shape[1]).float())
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == testy).sum().item()
    
        print('Test Accuracy: {} %'.format(100 * correct / total)) 
        
    figure()
    plt.plot(list(range(train_step)), Loss, color='green', label='Loss Function')
    plt.title('Loss Function')
    plt.legend()  # 显示图例

    #Random Forest        
    clf_grid = get_classifier_grid()
    X_shallow_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])    
    X_shallow_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])    

    clf_grid.fit(X_shallow_train, y_train)
    print_summary(clf_grid, X_shallow_test, y_test)
