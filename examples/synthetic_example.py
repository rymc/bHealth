import sys
sys.path.append('../')

import os
import pandas as pd
import numpy as np
from scipy import stats

from glob import glob

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.ma as ma

from datetime import datetime, timedelta

from digihealth.visualisations import plot_metrics
from digihealth.visualisations import features_figure
from digihealth.visualisations import plot_test_train_splits

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from digihealth.synthetic import RandomTimeSeries
#from synthetic import RandomTimeSeries
from digihealth.metric_wrappers import Wrapper

np.random.seed(42)

def get_raw_ts_X_y():
    def sin_gaussian_rand(n_samples, size):
        return np.sin(np.repeat(np.arange(n_samples), size).reshape(n_samples, size)
                        ) + np.random.randn(n_samples, size)

    n_features = 7
    features = ['Acceleration X', 'Acceleration Y', 'Acceleration Z']
    # First example
    generator_list = [lambda: np.random.randn(np.random.randint(1, 100), 3)/2
                                + [0, -1, 0],
                      lambda: np.random.randn(np.random.randint(10, 300), 3)/2
                                + [0, 0, -1],
                      lambda: sin_gaussian_rand(np.random.randint(15, 300), 3)
                                + [0, -1, 0],
                      lambda: sin_gaussian_rand(np.random.randint(30, 90), 3)*2
                                + [0, 0, -1],
                      lambda: np.random.randn(np.random.randint(5*60, 10*60), 3)/16
                                + [-0.5, 0.5, 0],
                      lambda: np.random.randn(np.random.randint(5 * 60, 10 * 60), 3) / 16
                              + [-0.5, 0.5, 0]
                     ]

    labels = [0, 1, 2, 3, 4, 5]
    rts = RandomTimeSeries(generator_list, labels=labels,
                           priors=[3, 4, 2, 1, 1, 1], samplesize='1Min')

    ts, X, y = rts.generate('06/02/2019', '13/02/2019')

    ts = ts.values

    return ts, X, y, labels, features


def preprocess_X_y(ts, X, y):
    return ts, X, y


def split_train_test(ts, X, y):
    # Create train and test partitions
    skf = StratifiedKFold(n_splits=2, shuffle=False)
    y = y.astype(int)
    train_index, test_index = skf.split(X, y).__next__()
    train = ts[train_index], X[train_index], y[train_index]
    test = ts[test_index], X[test_index], y[test_index]
    plot_test_train_splits(y[train_index], y[test_index])
    return train, test


def get_classifier_grid():
    # Create cross-validation partitions from training
    # This should select the best set of parameters
    cv = StratifiedKFold(n_splits=3, shuffle=False)

    models = {'rf': {'model': RandomForestClassifier(),
                     'parameters': {'n_estimators': [200],
                                    'min_samples_leaf': [1]}}
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


def generate_visualisations(clf, X, y, ts, labels, features):
    from digihealth.visualisations import labels_figure
    from digihealth.visualisations import features_figure
    y_pred = clf.predict(X)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(2, 1, 1)
    fig, ax = labels_figure(y_pred, ts=ts, labels=labels, fig=fig, ax=ax)
    ax = fig.add_subplot(2, 1, 2)
    df_X = pd.DataFrame(X, index=ts).resample('15Min').agg('mean')
    fig, ax = features_figure(df_X.values, ts=df_X.index, feature_names=features, fig=fig, ax=ax)
    fig.savefig('predictions.svg')

    # Ongoing example of polar plot
    from digihealth.visualisations import polar_labels_figure
    def most_common(x):
        if len(x) == 0:
            return -1
        return np.argmax(np.bincount(x))

    resample = '15Min'
    df_labels = pd.DataFrame(y_pred, columns=['label'], index=ts).resample(resample).agg(most_common)

    # Add NaN at the beginning for days before installation
    first_day = df_labels.index[0].replace(hour=0, minute=0, second=0,
                                         microsecond=0)
    previous_monday = first_day + timedelta(days=-first_day.weekday())
    leading_labels = (first_day - previous_monday) / resample
    df_labels = pd.concat([pd.DataFrame(data=-1, index=[previous_monday],
                                        columns=['label']),
                           df_labels])
    df_labels = df_labels.resample(resample).agg('first')
    df_labels = df_labels.fillna(-1)

    # Number of columns in one week
    n_columns = int(pd.Timedelta('1D') /
                     pd.Timedelta(resample))

    xticklabels = ('24', '1', '2', '3', '4', '5', '6', '7',
                   '8', '9', '10', '11', '12', '13', '14', '15',
                   '16', '17', '18', '19', '20', '21', '22', '23')
    filename = 'labels_daily.png'

    # Add carrying -1 (denoting NaNs)
    y_labels = df_labels['label'].values
    y_labels = np.concatenate((y_labels, -
                               np.ones(int(np.ceil(len(y_labels)/n_columns)*n_columns)
                                     - len(y_labels))))
    # Create a rectangular matrix with (number_weeks x labels_per_week)
    # Int this case every week has n_columns
    y_labels = y_labels.reshape((-1, n_columns)).astype(int)

    fig, ax = polar_labels_figure(y_labels, labels, xticklabels,
                                  empty_rows=4, leading_labels=0, spiral=True,
                                  title="{} per box".format(resample), m=None)
    fig.savefig(filename, dpi=300)

def activity_metrics(labels, timestamps, span):
    """Outputs typical activity metrics."""

    descriptor_map = {
        'eating': 0,
        'sitting': 1,
        'walking': 2,
        'studying': 3,
        'sleeping': 4,
        'washing': 5
    }

    metrics = Wrapper(labels, timestamps, span, 1, 25, descriptor_map, adjecency=None)

    df_time = timestamps.astype('datetime64')
    df_time = pd.DataFrame(df_time, columns=['Time'])
    df_label = pd.DataFrame(labels, columns=['Label'])

    metric_array = [metrics.duration_sitting,
                    metrics.duration_walking,
                    metrics.duration_washing,
                    metrics.duration_eating,
                    metrics.duration_sleeping,
                    metrics.duration_studying,
                    metrics.number_of_unique_activities]

    metric_container, date_container = metrics.run_metric_array(metric_array)

    return metric_container, date_container

if __name__ == '__main__':
    ts, X, y, labels, features = get_raw_ts_X_y()

    features_figure(X[0:X.size:50], ts[0:ts.size:50], feature_names=['X', 'Y', 'Z'])

    ts, X, y = preprocess_X_y(ts, X, y)
    (ts_train, X_train, y_train), (ts_test, X_test, y_test) = split_train_test(ts, X, y)
    clf_grid = get_classifier_grid()
    clf_grid.fit(X_train, y_train)
    print_summary(clf_grid, X_test, y_test)

    metric_container_daily, date_container_daily = activity_metrics(y, ts, 'daily')
    plot_metrics(metric_container_daily, date_container_daily)

    plt.show()
