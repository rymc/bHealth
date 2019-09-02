import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from digihealth.visualisations import plot_metrics
from digihealth.visualisations import features_figure
from digihealth.visualisations import plot_features
from digihealth.visualisations import plot_test_train_splits
from digihealth import data_loading
from digihealth import data_loading_debug

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from digihealth.transforms import Transforms
from digihealth.metrics import Metrics
from digihealth.metric_wrappers import Wrapper

def get_raw_ts_X_y():

    labels, ts, xyz = data_loading_debug.data_loader_accelerometer_debug()
    return ts, xyz, labels

def preprocess_X_y(ts, X, y):
    new_X = []
    new_y = []

    winlength_seconds = 10
    overlap_seconds = 1
    print("Window size of "+str(winlength_seconds)+" seconds and overlap of "+str(float(overlap_seconds) / winlength_seconds)+"%")
    samples_per_sec = 50
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
    # Remove datetime from features
    new_X = new_X[:, 1:]
    new_y = np.array(new_y)

    new_X = transform.feature_selection(new_X, new_y, 'tree')

    return new_X, new_y


def split_train_test(X, y):
    # Create train and test partitions
    skf = StratifiedKFold(n_splits=2, shuffle=False)
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
    clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True)
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

if __name__ == '__main__':
    ts, X, y = get_raw_ts_X_y()
    features_figure(X[0:X.size:50], feature_names=['X', 'Y', 'Z'])

    X, y = preprocess_X_y(ts, X, y)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)
    clf_grid = get_classifier_grid()
    clf_grid.fit(X_train, y_train)
    print_summary(clf_grid, X_test, y_test)

    metric_container_daily, date_container_daily = activity_metrics(y, ts, 'daily')
    plot_metrics(metric_container_daily, date_container_daily)

    plt.show()
