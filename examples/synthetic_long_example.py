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
from bhealth.visualisations import plot_metrics
from bhealth.visualisations import features_figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from bhealth.visualisations import plot_test_train_splits

from bhealth.synthetic import RandomTimeSeries
from bhealth.metric_wrappers import Wrapper


np.random.seed(42)

def get_raw_ts_X_y():
    def sin_gaussian_rand(n_samples, size):
        return np.sin(np.repeat(np.arange(n_samples), size).reshape(n_samples, size)
                        ) + np.random.randn(n_samples, size)

    n_features = 7
    features = ['rssi bathroom', 'rssi bedroom 1', 'rssi stairs', 'rssi hall',
                'rssi kitchen', 'rssi living room']
    generator_list = [
        lambda: np.random.randn(np.random.randint(2, 6), n_features)
                + [3, 1, 0, 1, 0, 0, 0],
        lambda: sin_gaussian_rand(np.random.randint(5*6, 8*6), n_features)
                + [0, 4, 1, 2, 1, 1, 0],
        lambda: np.random.randn(np.random.randint(3, 9), n_features)
                + [0, 1, 0, 5, 0, 0, 1],
        lambda: sin_gaussian_rand(np.random.randint(1, 9), n_features)
                + [1, 1, 0, 1, 2, 4, 3],
        lambda: np.random.randn(np.random.randint(3, 12), n_features)
                + [0, 1, 1, 0, 2, 3, 9],
                ]

    labels = ['bathroom', 'bedroom', 'hall', 'kitchen', 'living room']
    rts = RandomTimeSeries(generator_list, labels=labels,
                           priors=[5, 2, 4, 3, 1], samplesize='1Min')

    ts, X, y = rts.generate('01-01-2019', '08-01-2019')

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
    y_pred = clf.predict(X)
    # Ongoing example of polar plot
    from bhealth.visualisations import polar_labels_figure
    def most_common(x):
        if len(x) == 0:
            return -1
        return np.argmax(np.bincount(x))

    resample = '2H'
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

    n_columns *= 7
    xticklabels = ('Mon 00:00', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat',
                   'Sun')
    filename = 'labels_weekly.png'

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

def localisation_metrics(labels, timestamps, span):
    """Outputs typical activity metrics."""

    descriptor_map = {
        'bathroom' : [0],
        'bedroom 1' : [1],
        'bedroom 2' : [2],
        'kitchen' : [3],
        'living room' : [4]
    }

    adjecency = [[0, 2.5, 3, 3.3, 4],
                 [2.5, 0, 6, 1.5, 2],
                 [3, 6, 0, 4, 1],
                 [3.3, 1.5, 4, 0, 1],
                 [4, 3, 1, 1, 0]]

    if not os.path.exists('./output/'):
        os.mkdir('./output/')

    metrics = Wrapper(labels, timestamps, span, 1, 25, descriptor_map,
                      csv_prep=r'./output/localisation_metrics.csv',
                      adjecency=adjecency)

    df_time = timestamps.astype('datetime64')
    df_time = pd.DataFrame(df_time, columns=['Time'])
    df_label = pd.DataFrame(labels, columns=['Label'])

    metric_array= [ metrics.duration_in_bathroom,
                    metrics.duration_in_bedroom_1,
                    metrics.duration_in_bedroom_2,
                    metrics.duration_in_kitchen,
                    metrics.duration_in_living_room]

    metric_container, date_container = metrics.run_metric_array_csv(metric_array)

    return metric_container, date_container


if __name__ == '__main__':
    ts, X, y, labels, features = get_raw_ts_X_y()

    features_figure(X, ts, feature_names=['rssi bathroom', 'rssi bedroom 1',
                                          'rssi stairs', 'rssi hall',
                                          'rssi kitchen', 'rssi living room'])

    ts, X, y = preprocess_X_y(ts, X, y)
    (ts_train, X_train, y_train), (ts_test, X_test, y_test) = split_train_test(ts, X, y)
    clf_grid = get_classifier_grid()
    clf_grid.fit(X_train, y_train)
    print_summary(clf_grid, X_test, y_test)

    generate_visualisations(clf_grid.best_estimator_, X, y, ts, labels, features)

    metric_container_daily, date_container_daily = localisation_metrics(y, ts, 'hourly')
    plot_metrics(metric_container_daily, date_container_daily, labels_=labels)

    plt.savefig(__file__.strip('.py'))
