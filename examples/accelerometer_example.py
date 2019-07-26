import sys
sys.path.append('../')

import numpy as np
import pandas as pd
from digihealth import data_loading

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from digihealth.transforms import Transforms
from digihealth.metrics import Metrics

def get_raw_ts_X_y():

    labels, ts, xyz = data_loading.data_loader_accelerometer()
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
    return (X_train, y_train), (X_test, y_test)


def get_classifier_grid():
    # Create cross-validation partitions from training
    # This should select the best set of parameters
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    clf = RandomForestClassifier()
    param_grid = {'n_estimators' : [200, 250, 300],
                  'min_samples_leaf': [5, 10, 20, 40]}
    clf_grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True)
    return clf_grid


def print_summary(clf_grid, X_test, y_test):
    import numpy as np
    print('Best parameters are: {}'.format(clf_grid.best_params_))
    print("CV accuracy "+str(np.mean(clf_grid.cv_results_['mean_test_score'])))

    # The best model was fitted on the full training data, here it is tested only
    tt_score = clf_grid.score(X_test, y_test)
    print("Train / test split accuracy "+str(tt_score))


def activity_metrics(labels, timestamps):
    """Outputs typical activity metrics."""

    df_time = timestamps.astype('datetime64')
    df_time = pd.DataFrame(df_time, columns=['Time'])
    df_label = pd.DataFrame(labels, columns=['Label'])

    unique_days = df_time['Time'].dt.normalize().unique()
    for day in unique_days:
            hour = day
            print(hour)
            metric_container = {"timestamp": [], "metrics": []}
            for hr in range(23):
                hour_container = {}
                next_hour = hour + np.timedelta64(1, 'h')
                mask = ((df_time['Time'] > hour) & (df_time['Time'] <= next_hour))
                times = df_time.loc[mask]
                labs = df_label.loc[mask]

                if labs.size > 1:

                    times = times.astype(np.int64) // 10 ** 6
                    times = times / 1000

                    metr = Metrics(times, 3600, 1)

                    hourly_average_label_occurrence = metr.average_labels_per_window(labs, times)
                    hourly_average_location_stay = metr.duration_of_labels_per_window(labs, times)
                    hourly_average_number_of_changes = metr.number_of_label_changes_per_window(labs, times)
                    hourly_average_time_between_labels = metr.average_time_between_labels(labs, times)

                else:

                    hourly_average_label_occurrence = []
                    hourly_average_location_stay = []
                    hourly_average_number_of_changes =[]
                    hourly_average_time_between_labels = []

                hour_container["label_occurrence"] = hourly_average_label_occurrence
                hour_container["label_stay"] = hourly_average_location_stay
                hour_container["average_number_of_changes"] = hourly_average_number_of_changes
                hour_container["average_time_between_labels"] = hourly_average_time_between_labels

                metric_container["timestamp"].append(hour)
                metric_container["metrics"].append(hour_container)

                hour = next_hour

    return metric_container

if __name__ == '__main__':
    ts, X, y = get_raw_ts_X_y()
    X, y = preprocess_X_y(ts, X, y)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)
    clf_grid = get_classifier_grid()
    clf_grid.fit(X_train, y_train)
    print_summary(clf_grid, X_test, y_test)
    activity_metrics(y_test, ts)