from transforms import Transforms
import pandas as pd
import numpy as np

df = pd.read_csv('../data/realistic_sensor_displacement/subject2_ideal.log.gz', sep='\t', header=None)

acc= df.ix[:, [0, 68,69,70]]
acc.columns=['ts', 'x','y','z']
acc['ts'] = pd.to_datetime(acc['ts'], unit='s')

ts = acc[['ts']].values
xyz = acc[['x','y','z']].values
labels = df[df.columns[-1]].values

x = []
y = []

winlength_seconds = 3
overlap_seconds = 1
print("Window size of "+str(winlength_seconds)+" seconds and overlap of "+str(float(overlap_seconds) / winlength_seconds)+"%")
samples_per_sec = 50
winlength = samples_per_sec * winlength_seconds
current = winlength
overlap = samples_per_sec * overlap_seconds

transform = Transforms(window_length=winlength, window_overlap=overlap)
print("Use number of zero crossings, spectral entropy as features..")
feature_transforms = [transform.zero_crossings, transform.spec_entropy]

while True:
    windowed_raw = transform.slide(xyz)
    if len(windowed_raw) > 0:
        try:
            windowed_features = [ts[transform.current_position][0]]
        except Exception as e:
            break
        for function in feature_transforms:
            windowed_features.extend((np.apply_along_axis(function, 0, windowed_raw).tolist()))
        x.append(windowed_features)

        windowed_raw_labels = transform.slide(labels, update=False)
        most_freq_label = np.bincount(windowed_raw_labels).argmax()
        y.append(most_freq_label)

# Convert lists to Numpy arrays
x = np.array(x)
# Remove datetime from features
x = x[:, 1:]
y = np.array(y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, \
        GridSearchCV, train_test_split

# Create train and test partitions
skf = StratifiedKFold(n_splits=2, shuffle=False)
train_index, test_index = skf.split(x, y).__next__()
print(train_index)
print(test_index)
X_train, X_test = x[train_index], x[test_index]
y_train, y_test = y[train_index], y[test_index]

# Create cross-validation partitions from training
# This should select the best set of parameters
cv = StratifiedKFold(n_splits=5, shuffle=False)
clf = RandomForestClassifier()
param_grid = {'n_estimators' : [200, 250, 300],
              'min_samples_leaf': [5, 10, 20, 40]}
grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, refit=True)
grid.fit(X_train, y_train)
print('Best parameters are: {}'.format(grid.best_params_))
print("5 CV accuracy "+str(np.mean(grid.cv_results_['mean_test_score'])))


# The best model was fitted on the full training data, here it is tested only
tt_score = grid.score(X_test, y_test)
print("Train / test split accuracy "+str(tt_score))
