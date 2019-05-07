from transforms import Transforms
import pandas as pd
import numpy as np
import os

from datawrapper.house import HouseVisit
from datawrapper.utils import get_all_paths, df_start_end_to_time

datasets_path = '../data/houses'
all_bson_paths = get_all_paths(datasets_path, path_expression=r".*{0}(?P<hid>\w+){0}(?P<vid>\w+){0}sphere".format(os.sep))

visit = HouseVisit('9665', all_bson_paths[0])

df_items = visit.get_hypercat(os.path.join(visit.root, 'hypercat-db'))
df_ann = visit.get_annotations()

time_intervals = df_ann[df_ann['tier'] == 'Experiment'][['start', 'end']]
rssi_list = visit.get_rssi(time_intervals=time_intervals)

df_rssi = pd.concat(rssi_list)
try:
    href_loc_map = visit.get_href_locations()
except AttributeError as e:
    print(e)
df_rssi = df_rssi.resample('1S').agg(np.nanmean)
df_loc = df_ann[df_ann['tier'] == 'Location'][['start', 'end', 'label']]
df_loc = df_start_end_to_time(df_rssi.index, df_loc)
df_loc['label'] = df_loc['label'].cat.remove_unused_categories()

loc_mask = ~df_loc['label'].isna()
df_rssi = df_rssi[loc_mask]
df_loc = df_loc[loc_mask]

visit.y = df_loc['label'].cat.codes.values
visit.X = df_rssi.values

labels = df_loc['label'].cat.categories
visit.features = df_rssi.columns
visit.labels = df_loc['label'].cat.categories

x = visit.X
y = visit.y

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_predict
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

models = {'rf': {'model': RandomForestClassifier(),
                 'parameters': {'n_estimators': [200, 250, 300],
                                'min_samples_leaf': [1, 5, 10, 20]}},
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


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
classifier_name = 'rf'

steps = [('imputer', SimpleImputer(missing_values=np.nan,
                                   strategy='mean')),
         ('scaler', StandardScaler()),
         ('clf', models[classifier_name]['model'])]

pipeline = Pipeline(steps)

pipeline_parameters = {'clf__' + key: value for key, value in
                       models[classifier_name]['parameters'].items()}

grid = GridSearchCV(pipeline, param_grid=pipeline_parameters, cv=cv,
                    refit=True)

grid.fit(X_train, y_train)
print('Best parameters are: {}'.format(grid.best_params_))
print("5 CV accuracy "+str(np.mean(grid.cv_results_['mean_test_score'])))

# The best model was fitted on the full training data, here it is tested only
tt_score = grid.score(X_test, y_test)
print("Train / test split accuracy "+str(tt_score))
