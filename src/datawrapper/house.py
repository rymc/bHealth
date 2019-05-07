import os
import re
import gzip
import bson
import bz2
import pandas
from pandas.io.json import json_normalize
import numpy

from collections import Counter

def read_bson(file_path):
    '''Returns a json object from a Bson or zipped Bson file
    '''
    if file_path.endswith('.gz'):
        open_function = gzip.open
    elif file_path.endswith('.bz2'):
        open_function = bz2.open
    elif file_path.endswith('.bson'):
        open_function = open
    else:
        print('File extension unknown for {}'.format(file_path))
        return []

    with open_function(file_path, "rb") as f:
        json = bson.decode_all(f.read())

    return json


class HouseVisit(object):
    def __init__(self, uid, folder):
        self.uid = uid
        self.folder = folder
        self.db_folder = folder['root']
        self.root = os.path.dirname(self.db_folder)
        self.file_list = folder['file_list']

    def get_annotations(self, drop=None):
        '''Returns a pandas.DataFrame with all the annotations of the visit
        '''
        if hasattr(self, 'df_ann'):
            return self.df_ann

        # TODO add option to drop
        ann_file = list(filter(re.compile('ANN.bson*').match,
                               self.folder['file_list']))
        if len(ann_file) == 0:
            return pandas.DataFrame()
        ann = read_bson(os.path.join(self.db_folder, ann_file[0]))
        df_ann = pandas.DataFrame(ann)
        if len(df_ann) > 0:
            df_ann.set_index('_id', inplace=True)
            df_ann['hid'] = df_ann['hid'].astype('category')
            df_ann['label'] = df_ann['label'].astype('category')
            df_ann['tier'] = df_ann['tier'].astype('category')

        self.df_ann = df_ann
        return df_ann

    def experiments_report(self):
        if not hasattr(self, 'df_ann'):
            self.get_annotations()

        if len(self.df_ann) == 0:
            return pandas.DataFrame()

        if hasattr(self, 'df_exp'):
            return self.df_exp

        all_experiments = self.df_ann[self.df_ann['tier'] == 'Experiment']
        all_locations = self.df_ann[self.df_ann['tier'] == 'Location']['label'].unique()
        rows = []
        for i, experiment in enumerate(all_experiments.iterrows()):
            row = [i+1]
            row.extend((experiment[1]['start'], experiment[1]['end']))
            mask = ((self.df_ann['tier'] == 'Location') &
                    (self.df_ann['start'].between(experiment[1]['start'],
                                            experiment[1]['end'])) &
                    (self.df_ann['end'].between(experiment[1]['start'],
                                          experiment[1]['end'])))
            exp_loc = self.df_ann[mask]['label'].unique()
            row.extend([loc in exp_loc for loc in all_locations])
            rows.append(row)

        columns = ['Experiment', 'start', 'end']
        columns.extend( all_locations.astype(str))
        df_exp = pandas.DataFrame(rows, columns=columns)
        df_exp.set_index('Experiment', inplace=True)

        self.df_exp = df_exp
        return df_exp

    def get_hypercat(self, folder):
        '''Returns a pandas.DataFrame with hypercat
        '''
        if hasattr(self, 'df_items'):
            return self.df_items

        items_file = os.path.join(folder, 'hypercat-db', 'hypercat', 'items.bson')
        if not os.path.isfile(items_file):
            return pandas.DataFrame()
        items = read_bson(items_file)
        df_items = pandas.DataFrame(items)
        if len(df_items) > 0:
            df_items.set_index('_id', inplace=True)

        def get_field(array, name):
            for dictionary in array:
                if dictionary['rel'] == name:
                    return dictionary['val']
            return None

        def get_location(array):
            return get_field(array, 'house_location')

        def get_device_type(array):
            return get_field(array, 'device_type')

        def get_description(array):
            return get_field(array, 'urn:X-tsbiot:rels:hasDescription:en')

        def get_appliance(array):
            return get_field(array, 'appliance')

        column_map = {'house_location': get_location,
                      'appliance': get_appliance,
                      'description': get_description,
                      'device_type': get_device_type}

        for key, function in column_map.items():
            df_items[key] = df_items['i-object-metadata'].apply(function)

        def NUC_mac_to_mac(string):
            return ':'.join(string[i:i+2].lower() for i in range(0, len(string), 2))

        mask = df_items['device_type'].str.startswith('Intel NUC')
        df_items.loc[mask, 'href'] = df_items[mask]['href'].apply(NUC_mac_to_mac)

        self.df_items = df_items
        return df_items

    def get_wearable(self, time_intervals=None):
        '''Returns the wearable data from the given intervals of time
        '''
        wear_file = list(filter(re.compile('WEAR.bson*').match,
                                self.folder['file_list']))
        if len(wear_file) == 0:
            return None
        wear = read_bson(os.path.join(self.db_folder, wear_file[0]))
        df_wear = pandas.DataFrame(wear)
        if '_id' not in df_wear.columns:
            return []

        df_wear.set_index('_id', inplace=True)
        df_wear['hid'] = df_wear['hid'].astype('category')
        df_wear['uid'] = df_wear['uid'].astype('category')
        if time_intervals is None:
            return [df_wear]

        wear_list = []
        for ti in time_intervals.iterrows():
            wear_list.append(df_wear[df_wear['bt'].between(
                                    ti[1]['start'], ti[1]['end'])])
        return wear_list


    def get_rssi(self, time_intervals=None):
        '''Returns the wearable data from the given intervals of time
        '''
        wear_file = list(filter(re.compile('WEAR.bson*').match,
                                self.folder['file_list']))
        if len(wear_file) == 0:
            return None
        wear = read_bson(os.path.join(self.db_folder, wear_file[0]))
        result = json_normalize(wear, 'gw', ['bt', 'uid', 'ts'],
                                meta_prefix='wear_', errors='ignore')
        try:
            df_rssi = result.pivot_table(index='wear_bt', columns='uid',
                                         values='rssi')
        except KeyError as e:
            print(e)
            return []

        if time_intervals is None:
            return [df_rssi]

        rssi_list = []
        for ti in time_intervals.iterrows():
            rssi_list.append(df_rssi.loc[ti[1]['start']:ti[1]['end']])
        return rssi_list

    def get_wear_uid_counts(self):
        '''Returns the wearable data from the given intervals of time
        '''
        wear_file = list(filter(re.compile('WEAR.bson*').match,
                                self.folder['file_list']))
        if len(wear_file) == 0:
            return None
        wear = read_bson(os.path.join(self.db_folder, wear_file[0]))
        uid_counts = Counter()
        for w in wear:
            for field in w['gw']:
                uid_counts[field['uid']] += 1

        return uid_counts

    def get_wear_uid_bts(self):
        '''Returns the wearable data from the given intervals of time
        '''
        wear_file = list(filter(re.compile('WEAR.bson*').match,
                                self.folder['file_list']))
        if len(wear_file) == 0:
            return None
        wear = read_bson(os.path.join(self.db_folder, wear_file[0]))
        uid_bts = {}
        for w in wear:
            for field in w['gw']:
                if field['uid'] not in uid_bts:
                    uid_bts[field['uid']] = []
                uid_bts[field['uid']].append(w['bt'])

        return uid_bts

    def get_env_uid_counts(self):
        '''Returns the wearable data from the given intervals of time
        '''
        env_file = list(filter(re.compile('ENV.bson*').match,
                                self.folder['file_list']))
        if len(env_file) == 0:
            return None
        env = read_bson(os.path.join(self.db_folder, env_file[0]))
        uid_counts = Counter()
        for e in env:
            uid_counts[e['uid']] += 1

        return uid_counts

    def get_env_uid_bts(self):
        '''Returns the environmental sensors packets
        '''
        env_file = list(filter(re.compile('ENV.bson*').match,
                                self.folder['file_list']))
        if len(env_file) == 0:
            return None
        env = read_bson(os.path.join(self.db_folder, env_file[0]))
        uid_bts = {}
        for e in env:
            if e['uid'] not in uid_bts:
                uid_bts[e['uid']] = []
            uid_bts[e['uid']].append(e['bt'])

        return uid_bts

    def get_href_locations(self):
        if self.df_items is None:
            return {}
        self.href_locations = {href: '\n'.join((hl, dt)) for href, hl, dt in
                               self.df_items[['href', 'house_location',
                                              'device_type']].values.astype(str)}
        return self.href_locations

    def predict_location(self, visit, probability=False):
        X = numpy.ones((visit.X.shape[0], self.features.shape[0]))
        # Impute mean for missing features
        X *= self.X.mean(axis=0)
        for i, key in enumerate(self.features):
            where = numpy.where(key == visit.features)[0]
            if len(where) == 1:
                X[:, i] = visit.X[:, where].flatten()
        if probability:
            return self.clf.predict_proba(X)
        return self.clf.predict(X)

    def number_times_without_rssi(self, df_rssi, resample='5Min'):
        df_resampled = df_rssi.resample(resample).agg(numpy.mean)
        df_nans = df_resampled.isna().all(axis=1)
        times_wo_rssi = (df_nans & (df_nans !=
                                    df_nans.shift(1))).cumsum().max()
        return times_wo_rssi
