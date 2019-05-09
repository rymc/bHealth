def get_raw_ts_X_y():
    import pandas as pd
    df = pd.read_csv('../data/realistic_sensor_displacement/subject2_ideal.log.gz', sep='\t', header=None)

    acc= df.ix[:, [0, 68,69,70]]
    acc.columns=['ts', 'x','y','z']
    acc['ts'] = pd.to_datetime(acc['ts'], unit='s')

    ts = acc[['ts']].values
    xyz = acc[['x','y','z']].values
    labels = df[df.columns[-1]].values
    return ts, xyz, labels


def preprocess_X_y(ts, X, y):
    import numpy as np
    from transforms import Transforms
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
    print("Use number of mean crossings, spectral entropy as features..")
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
    from sklearn.model_selection import StratifiedKFold
    # Create train and test partitions
    skf = StratifiedKFold(n_splits=2, shuffle=False)
    train_index, test_index = skf.split(X, y).__next__()
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    return (X_train, y_train), (X_test, y_test)


def get_classifier_grid():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
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


if __name__ == '__main__':
    ts, X, y = get_raw_ts_X_y()
    X, y = preprocess_X_y(ts, X, y)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)
    clf_grid = get_classifier_grid()
    clf_grid.fit(X_train, y_train)
    print_summary(clf_grid, X_test, y_test)
