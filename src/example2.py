import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

from .datawrapper.house import HouseVisit
from .datawrapper.utils import get_all_paths, df_start_end_to_time


def get_raw_ts_X_y():

    datasets_path = '../data/houses'
    path_expression = r".*{0}(?P<hid>\w+){0}(?P<vid>\w+){0}sphere".format(
        os.sep)
    all_bson_paths = get_all_paths(datasets_path,
                                   path_expression=path_expression)

    visit = HouseVisit('9665', all_bson_paths[0])

    df_ann = visit.get_annotations()

    time_intervals = df_ann[df_ann['tier'] == 'Experiment'][['start', 'end']]
    rssi_list = visit.get_rssi(time_intervals=time_intervals)

    df_rssi = pd.concat(rssi_list)
    df_rssi = df_rssi.resample('1S').agg(np.nanmean)
    df_loc = df_ann[df_ann['tier'] == 'Location'][['start', 'end', 'label']]
    df_loc = df_start_end_to_time(df_rssi.index, df_loc)
    df_loc['label'] = df_loc['label'].cat.remove_unused_categories()

    loc_mask = ~df_loc['label'].isna()
    df_rssi = df_rssi[loc_mask]
    df_loc = df_loc[loc_mask]

    visit.y = df_loc['label'].cat.codes.values
    visit.X = df_rssi.values

    visit.features = df_rssi.columns
    visit.labels = df_loc['label'].cat.categories

    X = visit.X
    y = visit.y
    return (), X, y


def preprocess_X_y(ts, X, y):
    return X, y


def split_train_test(X, y):
    # Create train and test partitions
    skf = StratifiedKFold(n_splits=2, shuffle=False)
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


if __name__ == '__main__':
    ts, X, y = get_raw_ts_X_y()
    X, y = preprocess_X_y(ts, X, y)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)
    clf_grid = get_classifier_grid()
    clf_grid.fit(X_train, y_train)
    print_summary(clf_grid, X_test, y_test)
