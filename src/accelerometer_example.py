import numpy as np
import numpy.ma as ma
import numpy.matlib
import pandas as pd
from datetime import datetime
from datetime import timedelta

from glob import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.model_selection import StratifiedKFold

from transforms import Transforms
from metrics import Metrics

def data_loader():

    global xyz_
    data_directory = '../data/acc_loc_data/experiments/*/'

    folders = glob(data_directory)

    ts = np.array([[]]).reshape(0, 1)
    xyz = np.array([[]]).reshape(0, 3)
    labels = np.array([[]]).reshape(0, 1)

    print('Found', len(folders), 'experiment folders.')

    for idx, fold in enumerate(folders):

        if idx == 0 :

            print('Running folder: ', idx + 1)

            fold_data = [fold + 'accelerometer_filtered.csv']
            fold_data = ''.join(fold_data)

            data = pd.read_csv(fold_data, skiprows=1, usecols=[0, 4, 5, 6])

            acc = data
            acc.columns = ['ts', 'x', 'y', 'z']
            acc['ts'] = pd.to_datetime(acc['ts'])

            ts_ = acc[['ts']].values

            xyz_ = acc[['x', 'y', 'z']].values

            fold_labels = [fold + 'activity_annotation_times.csv']
            fold_labels = ''.join(fold_labels)

            labs = pd.read_csv(fold_labels)

            labels_ = np.zeros((ts_.size,), dtype=int)

            for lab in labs.itertuples():
                start_date = pd.to_datetime(lab.timestamp_start)
                end_date = pd.to_datetime(lab.timestamp_end)
                label_ = lab.activity_tag
                mask = (ts_ > start_date) & (ts_ <= end_date)

                for idx_mask, mk in enumerate(mask):
                    if mk == True:
                        labels_[idx_mask] = int(label_)

            for idx, t in enumerate(ts_):
                times = t[0].timestamp()
                local = datetime.fromtimestamp(times)
                ts_[idx] = local.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

            labels = np.append(labels, labels_)
            ts = np.concatenate((ts, ts_), axis=0)
            xyz = np.concatenate((xyz, xyz_), axis=0)

        labels = labels.astype(int)

    return labels, ts, xyz


def get_raw_ts_X_y():

    labels, ts, xyz = data_loader()
    return ts, xyz, labels


def preprocess_X_y(ts, X, y):
    new_X = []
    new_y = []

    winlength_seconds = 3
    overlap_seconds = 1
    print("Window size of "+str(winlength_seconds)+" seconds and overlap of "+str(float(overlap_seconds) / winlength_seconds)+"%")
    samples_per_sec = 50
    winlength = samples_per_sec * winlength_seconds
    current = winlength
    overlap = samples_per_sec * overlap_seconds

    transform = Transforms(window_length=winlength, window_overlap=overlap)
    print("Use number of mean crossings, spectral entropy as features...")
    feature_transforms = [transform.mean_crossings, transform.spec_entropy]

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
        next_day = day + np.timedelta64(1, 'D')
        mask = ((df_time['Time'] > day) & (df_time['Time'] <= next_day))
        times = df_time.loc[mask]
        labs = df_label.loc[mask]

        if labs.size > 1:

            # For now, I cast it into posix time to make the metrics easier to analyse. This is because
            # they assume that the timestamp is in seconds // MK

            times = times.astype(np.int64) // 10 ** 6
            times = times / 1000

            metr = Metrics(times, 86400, 1)
            daily_average_label_occurence = metr.average_labels_per_window(labs, times)
            daily_average_location_stay = metr.duration_of_labels_per_window(labs, times)
            daily_average_number_of_changes = metr.number_of_label_changes_per_window(labs, times)
            daily_average_time_between_labels = metr.average_time_between_labels(labs, times)

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

                    metr = Metrics(times, 3600, 1)
                    hourly_average_label_occurence = metr.average_labels_per_window(labs, times)
                    hourly_average_location_stay = metr.duration_of_labels_per_window(labs, times)
                    hourly_average_number_of_changes = metr.number_of_label_changes_per_window(labs, times)
                    hourly_average_time_between_labels = metr.average_time_between_labels(labs, times)

                hour = next_hour

if __name__ == '__main__':
    ts, X, y = get_raw_ts_X_y()
    X, y = preprocess_X_y(ts, X, y)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)
    clf_grid = get_classifier_grid()
    clf_grid.fit(X_train, y_train)
    print_summary(clf_grid, X_test, y_test)
    activity_metrics(y_test, ts)
