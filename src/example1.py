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


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

clf = RandomForestClassifier(n_estimators=250,)
cv_score = cross_val_score(clf, x, y, cv=5)
print("5 CV accuracy "+str(np.mean(cv_score)))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=0)
clf = RandomForestClassifier(n_estimators=250,)
clf.fit(X_train, y_train)
tt_score = clf.score(X_test, y_test)
print("Train / test split accuracy "+str(tt_score))
