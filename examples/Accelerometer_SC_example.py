"""
This is an example dealing with SPHERE challenge data.
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
from digihealth.visualisations import plot_metrics
from digihealth.visualisations import features_figure
from digihealth.visualisations import plot_features
from digihealth.visualisations import plot_test_train_splits
from digihealth import data_loading
from digihealth.transforms import Transforms
from digihealth.metrics import Metrics
from digihealth.metric_wrappers import Wrapper

import scipy.stats
from scipy.signal import butter, filtfilt
from visualise_data import Sequence
from visualise_data import SequenceVisualisation

SCHLNG_data_path = '../SCHLNG_data'
metadata_path = '../SCHLNG_data/metadata'    

def load_sequence(file_id):
    filename = str(file_id).zfill(5)

    df = pd.read_csv('{}/train/{}/columns_1000ms.csv'.format(SCHLNG_data_path, filename))
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
            df.to_csv('{}/{}/{}/columns_1000ms.csv'.format(SCHLNG_data_path, train_test, stub_name), index=False)
    
            if train_test is 'train' or np.mod(fi, 50) == 0:
                if train_test is 'train': print 
                print ("Finished feature extraction for {}/{}\n".format(train_test, stub_name))
                
    column_names            
    
    # Load the training and testing data 
    train_x, train_y = load_sequences([1, 2, 3, 4, 5])
    
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

def preprocess_X_y(ts, X, y):
    new_X = []
    new_y = []

    winlength_seconds = 10
    overlap_seconds = 1
    print("Window size of "+str(winlength_seconds)+" seconds and overlap of "+str(float(overlap_seconds) / winlength_seconds)+"%")
    samples_per_sec = 1
    winlength = samples_per_sec * winlength_seconds
    current = winlength
    overlap = samples_per_sec * overlap_seconds

    transform = Transforms(window_length=winlength, window_overlap=overlap)
    print("Use number of mean crossings, spectral entropy as features...")
    feature_transforms = [transform.mean_crossings,
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
                windowed_features = ts[transform.current_position].split()
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
    # Remove datetime from features
    new_X = new_X[:, 1:]
    new_y = np.array(new_y)
    new_X = transform.feature_selection(new_X, new_y, 'tree')

    return new_X, new_y

def split_train_test(X, y):
    # Create train and test partitions
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    train_index, test_index = skf.split(X, y).__next__()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return (X_train, y_train), (X_test, y_test)


def get_classifier_grid():
#    Create cross-validation partitions from training
#    This should select the best set of parameters
#    cv = StratifiedKFold(n_splits=5, shuffle=False)
#    clf = RandomForestClassifier()
#    param_grid = {'n_estimators' : [200, 250],
#                  'min_samples_leaf': [5, 10]}
#    clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True)
#    return clf_grid
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


def activity_metrics(labels, timestamps, span):
    """Outputs typical activity metrics."""

    descriptor_map = {
                'sitting' : 77,
                'walking' : 78,
                'washing' : 79,
                'eating'  : 80,
                'sleeping': 81,
                'studying': 82
            }

    metrics = Wrapper(labels, timestamps, span, 1, 25, descriptor_map, adjecency=None)

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

    metric_container, date_container = metrics.run_metric_array(metric_array)
    return metric_container, date_container


def brier_score(given, predicted, weight_vector): 
    return np.power(given - predicted, 2.0).dot(weight_vector).mean()

if __name__ == '__main__':
    ts, X, y = get_raw_ts_X_y()
    X, y = preprocess_X_y(ts, X, y)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)
    clf_grid = get_classifier_grid()
    clf_grid.fit(X_train, y_train)
    y_prob = clf_grid.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)    
    cfm = confusion_matrix(y_test, y_pred)    
    class_weights = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    y_test_prob = np.zeros((y_test.shape[0], max(y_train)+1),np.float64) 
    for idx in range(0, y_test.shape[0]):
        y_test_prob[idx,y_test[idx]] = 1
    bs = brier_score(y_test_prob, y_prob, class_weights)
    print_summary(clf_grid, X_test, y_test)  