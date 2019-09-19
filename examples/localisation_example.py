import os
import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from bhealth import data_loading_debug

from bhealth.visualisations import plot_metrics
from bhealth.visualisations import features_figure
from bhealth.visualisations import features_figure_scatter
from bhealth.visualisations import plot_test_train_splits

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

from bhealth.metrics import Metrics
from bhealth.transforms import Transforms
from bhealth.metric_wrappers import Wrapper

def get_raw_ts_X_y(house_):

    ts, X, y = data_loading_debug.data_loader_rssi_debug(house_)
    return ts, X, y


def preprocess_X_y(ts, X, y):

    return X, y


def split_train_test(X, y):
    # Create train and test partitions
    skf = StratifiedKFold(n_splits=2, shuffle=False)
    y = y.astype(int)
    train_index, test_index = skf.split(X, y).__next__()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # plot_test_train_splits(y_train, y_test)
    return (X_train, y_train), (X_test, y_test)


def get_classifier_grid():
    # Create cross-validation partitions from training
    # This should select the best set of parameters
    cv = StratifiedKFold(n_splits=5, shuffle=False)

    models = {'rf': {'model': RandomForestClassifier(),
                     'parameters': {'n_estimators': [200, 250],
                                    'min_samples_leaf': [1, 5, 10]}},
              'lr': {'model': LogisticRegression(penalty='l2'),
                     'parameters': {'C': [0.01, 0.1, 1, 10, 100]}},
              'svc': {'model': SVC(probability=True),
                      'parameters': [{'kernel': ['rbf'],
                                      'gamma': [1e-3, 1e-4],
                                      'C': [1, 10, 100, 1000]},
                                     {'kernel': ['linear'],
                                      'C': [1, 10, 100, 1000]}]},
              'svc-rbf': {'model': SVC(probability=True),
                          'parameters': [{'kernel': ['rbf'],
                                          'gamma': [1e-3, 1e-4],
                                          'C': [1, 10, 100, 1000]}, ]},
              }

    classifier_name = 'rf'

    steps = [('imputer', SimpleImputer(missing_values=np.nan,
                                       strategy='mean')),
             ('scaler', StandardScaler()),
             ('clf', models[classifier_name]['model'])]

    pipeline = Pipeline(steps)

    pipeline_parameters = {'clf__' + key: value for key, value in
                           models[classifier_name]['parameters'].items()}

    clf_grid = GridSearchCV(pipeline, param_grid=pipeline_parameters, cv=cv,
                            refit=True, iid=True)
    return clf_grid


def print_summary(clf_grid, X_test, y_test):
    print('Best parameters are: {}'.format(clf_grid.best_params_))
    print("CV accuracy "+str(np.mean(clf_grid.cv_results_['mean_test_score'])))

    # The best model was fitted on the full training data, here tested only
    tt_score = clf_grid.score(X_test, y_test)
    print("Train / test split accuracy "+str(tt_score))

def localisation_metrics(labels, timestamps, span):
    """Outputs typical activity metrics."""

    descriptor_map = {
        'foyer' : [0],
        'bedroom' : [1],
        'living_room' : [2],
        'bathroom' : [3]
    }

    adjecency = [[0, 2.5, 3, 3.3],
                 [2.5, 0, 6, 1.5],
                 [3, 6, 0, 4],
                 [3.3, 1.5, 4, 0]]

    if not os.path.exists('./output/'):
        os.mkdir('./output/')

    metrics = Wrapper(labels, timestamps, span, 1, 25, descriptor_map,
                      adjecency=adjecency)

    df_time = timestamps.astype('datetime64')
    df_time = pd.DataFrame(df_time, columns=['Time'])
    df_label = pd.DataFrame(labels, columns=['Label'])

    metric_array= [metrics.walking_speed,
                   metrics.room_transfers,
                   metrics.number_of_unique_locations]

    metric_container, date_container = metrics.run_metric_array(
        metric_array, csv='./output/localisation.csv')

    return metric_container, date_container

if __name__ == '__main__':
    houses = ['A']

    for house_ in houses:

        ts, X, y = get_raw_ts_X_y(house_)
        X, y = preprocess_X_y(ts, X, y)
        (X_train, y_train), (X_test, y_test) = split_train_test(X, y)
        clf_grid = get_classifier_grid()
        clf_grid.fit(X_train, y_train)
        print_summary(clf_grid, X_test, y_test)

        metric_container_daily, date_container_daily = localisation_metrics(y,
                                                                            ts,
                                                                            'daily')
        figures_dict = plot_metrics(metric_container_daily,
                                    date_container_daily, labels_ = ['foyer',
                                                                     'bedroom',
                                                                     'living_room',
                                                                     'bathroom'])

        fig, ax = features_figure(X, feature_names=['AP1', 'AP2', 'AP3', 'AP4',
                                                    'AP5', 'AP6', 'AP7',
                                                    'AP8'])
        figures_dict['features'] = fig

        for key, fig in figures_dict.items():
            fig.savefig(os.path.join('./output/', key))
