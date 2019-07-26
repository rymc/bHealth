import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from digihealth import data_loading

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

from digihealth.metrics import Metrics
from digihealth.transforms import Transforms

def get_raw_ts_X_y(house_):

    ts, X, y = data_loading.data_loader_rssi(house_)
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
                            refit=True)
    return clf_grid


def print_summary(clf_grid, X_test, y_test):
    print('Best parameters are: {}'.format(clf_grid.best_params_))
    print("CV accuracy "+str(np.mean(clf_grid.cv_results_['mean_test_score'])))

    # The best model was fitted on the full training data, here tested only
    tt_score = clf_grid.score(X_test, y_test)
    print("Train / test split accuracy "+str(tt_score))

def localisation_metrics(labels, timestamps):
    """Outputs typical localisation metrics."""

    df_time = timestamps.astype('datetime64')
    df_time = pd.DataFrame(df_time, columns=['Time'])
    df_label = pd.DataFrame(labels, columns=['Label'])

    unique_days = df_time['Time'].dt.normalize().unique()
    for day in unique_days:
        next_day = day + np.timedelta64(1,'D')
        mask = ((df_time['Time'] > day) & (df_time['Time'] <= next_day))
        times = df_time.loc[mask]
        labs = df_label.loc[mask]

        if labs.size > 1:

            #For now, I cast it into posix time to make the metrics easier to analyse. This is because
            #they assume that the timestamp is in seconds // MK

            times = times.astype(np.int64) // 10**6
            times = times / 1000

            metr = Metrics(times, 86400, 1, 25)
            daily_average_label_occurence = metr.average_labels_per_window(labs, times)
            daily_average_location_stay = metr.duration_of_labels_per_window(labs, times)
            daily_average_number_of_changes= metr.number_of_label_changes_per_window(labs, times)
            daily_average_time_between_labels= metr.average_time_between_labels(labs, times)

            hour = day
            for hr in range(23):
                next_hour = hour + np.timedelta64(1, 'h')
                mask = ((df_time['Time'] > hour) & (df_time['Time'] <= next_hour))
                times = df_time.loc[mask]
                labs = df_label.loc[mask]

                if labs.size > 1:

                    # For now, I cast it into posix time to make the metrics easier to analyse. This is because
                    # they assume that the timestamp is in seconds // MK

                    times = times.astype(np.int64) // 10 ** 6
                    times = times / 1000

                    metr = Metrics(times, 3600, 1, 25)
                    hourly_average_label_occurence = metr.average_labels_per_window(labs, times)
                    hourly_average_location_stay = metr.duration_of_labels_per_window(labs, times)
                    hourly_average_number_of_changes = metr.number_of_label_changes_per_window(labs, times)
                    hourly_average_time_between_labels = metr.average_time_between_labels(labs, times)

                hour = next_hour


    # TODO Number of times bathroom visited during the night //MK
    # TODO Number of times kitchen visited during the night //MK

if __name__ == '__main__':
    houses = ['A', 'B', 'C', 'D']

    for house_ in houses:
        ts, X, y = get_raw_ts_X_y(house_)
        X, y = preprocess_X_y(ts, X, y)
        (X_train, y_train), (X_test, y_test) = split_train_test(X, y)
        clf_grid = get_classifier_grid()
        clf_grid.fit(X_train, y_train)
        print_summary(clf_grid, X_test, y_test)
        localisation_metrics(y_test, ts)

