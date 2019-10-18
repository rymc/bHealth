"""
data_loading.py
====================================
File with data loading functions.
"""

import numpy as np
import pandas as pd
import datetime
import csv

import os

from scipy import stats
from datetime import datetime
from glob import glob

def data_loader_accelerometer(house_id=None):
    """
    Load example accelerometer data.

    house_id: list of strings (eg. ['A', 'B']
    """

    data_directory = os.path.join('..', 'data', 'acc_loc_data', 'ble-accelerometer-indoor-localisation-measurements', 'house*{}'.format(os.path.sep))

    ts = np.array([[]]).reshape(0, 1)
    ts_n = np.array([[]]).reshape(0, 1)
    xyz = np.array([[]]).reshape(0, 3)
    labels = np.array([[]]).reshape(0, 1)

    folders = glob(data_directory)

    print('Found', len(folders), 'house folders.')

    for idx_house, fold_house in enumerate(folders):

        this_house_id = fold_house[-2]

        if (house_id is not None) and (this_house_id not in house_id):
            continue

        meta_data = [fold_house + 'metadata/act_desc.dat']
        meta_data = ''.join(meta_data)

        data = pd.read_csv(meta_data, header=None)

        experiment_folders = glob(os.path.join(fold_house, 'experiments/living*/'))

        print('Found', len(experiment_folders), 'experiment folders.')
        

        for idx, fold in enumerate(experiment_folders):

            print('Running folder: ', idx + 1)

            fold_data = [fold + 'accelerometer_filtered.dat']
            fold_data = ''.join(fold_data)

            data = pd.read_csv(fold_data, skiprows=1, usecols=[0, 4, 5, 6])

            acc = data
            acc.columns = ['ts', 'x', 'y', 'z']
            acc['ts'] = pd.to_datetime(acc['ts'])

            ts_ = acc[['ts']].values
            ts_n_ = acc[['ts']].values

            xyz_ = acc[['x', 'y', 'z']].values

            fold_labels = [fold + 'activity_annotation_times.dat']
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
                ts_n_[idx] = times
                ts_[idx] = local.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

            labels = np.append(labels, labels_)
            ts_n = np.concatenate((ts_n, ts_n_), axis=0)
            ts = np.concatenate((ts, ts_), axis=0)
            xyz = np.concatenate((xyz, xyz_), axis=0)

            labels = labels.astype(int)
            ts_n = ts_n.astype(float)

    sort_index = np.argsort(ts_n, axis=None)

    labels = labels[sort_index]
    ts = ts[sort_index]
    xyz = xyz[sort_index]

    return labels, ts, xyz

def data_loader_rssi(house_):
    """
     Load example RSSI data.
     """

    window = 10  # in seconds

    data_directory = ('../data/acc_loc_data/ble-accelerometer-indoor-localisation-measurements/house_' + house_ + '/')

    folders = glob(data_directory)

    ts = np.array([]).reshape(0, 1)
    X = []
    y = []

    print('Found', len(folders), 'house folders.')

    for idx_house, fold_house in enumerate(folders):

        meta_data_cord = [fold_house + 'metadata/tag_coordinates.dat']
        meta_data_cord = ''.join(meta_data_cord)
        data_coords = pd.read_csv(meta_data_cord, header=None)

        meta_data_adj = [fold_house + 'metadata/tag_adjacency.dat']
        meta_data_adj = ''.join(meta_data_adj)

        # with open(meta_data_adj) as f:
        #     reader = csv.reader(f)
        #     for row in reader:
        #         break

        experiment_folders = glob(fold_house + 'experiments/living*/')

        print('Found', len(experiment_folders), 'experiment folders.')

        for idx, fold in enumerate(experiment_folders):

            print('Running folder: ', idx + 1)

            fold_data = [fold + 'rx_wearable_data.dat']
            fold_data = ''.join(fold_data)

            data = pd.read_csv(fold_data, skiprows=1, usecols=[0, 1, 3])

            acc = data
            acc.columns = ['ts', 'ap', 'rssi']
            acc['ts'] = pd.to_datetime(acc['ts'])

            ts_ = acc[['ts']].values
            rssi_ = acc[['rssi']].values
            ap_ = acc[['ap']].values

            unique_aps_ = np.unique(ap_)

            for idx_ts, t in enumerate(ts_):
                ts_[idx_ts] = t[0].timestamp()

            fold_labels = [fold + 'tag_annotations.dat']
            fold_labels = ''.join(fold_labels)

            tag_data = pd.read_csv(fold_labels, skiprows=1, usecols=[0, 2])

            labs = tag_data
            labs.columns = ['ts', 'tag']
            labs['ts'] = pd.to_datetime(labs['ts'])

            ts_lab_ = labs[['ts']].values
            tag_ = labs[['tag']].values

            for idx_ts_lab, t in enumerate(ts_lab_):
                ts_lab_[idx_ts_lab] = t[0].timestamp()

            min_rss = ts_.min()
            min_lab = ts_lab_.min()
            max_rss = ts_.max()
            max_lab = ts_lab_.max()

            if min_rss > min_lab:
                first_ts = ts_lab_[0]
            else:
                first_ts = ts_[0]

            if max_rss > max_lab:
                last_ts = ts_lab_[-1]
            else:
                last_ts = ts_[-1]

            permutable = int((last_ts - first_ts) / window)

            rolling_index = first_ts

            for idx_sync in range(permutable):
                X_inter = []
                y_inter = []
                ts_inter = []
                range_min = rolling_index
                range_max = rolling_index + window
                ts_inter.append(range_min)
                for ap_idx in unique_aps_:
                    tag_masked = tag_[(ts_lab_ > range_min) & (ts_lab_ <= range_max)]

                    if tag_masked.size != 0:
                        y_inter.append(stats.mode(tag_masked)[0])
                        rssi_masked = rssi_[(ts_ > range_min) & (ts_ <= range_max)]
                        ap_masked = ap_[(ts_ > range_min) & (ts_ <= range_max)]
                        rssi_masked = rssi_masked[ap_masked == ap_idx]

                        if rssi_masked.size != 0:
                            X_inter.append(np.mean(rssi_masked))
                        else:
                            X_inter.append(-120)
                    else:
                        X_inter.append(-120)
                        y_inter.append(0)

                X.append(X_inter)
                y.append(stats.mode(y_inter)[0])
                ts = np.concatenate((ts, ts_inter), axis=0)
                rolling_index = range_max

    X = np.array(X)
    y = np.array(y)
    y = np.squeeze(y)

    sort_index = np.argsort(ts, axis=None)

    y = y[sort_index]
    ts = ts[sort_index]
    X = X[sort_index]

    for idx, t in enumerate(ts):
        local = datetime.fromtimestamp(t)
        ts[idx] = local.strftime("%Y-%m-%dT%H:%M:%S.%f%z")

    descriptor_map = {
        'foyer': [0, 1],
        'bedroom': [2, 3, 4, 5, 6, 7, 8, 79, 81],
        'living_room': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 77, 78, 82],
        'bathroom': [21, 22, 80]
    }

    labels_ = np.zeros([len(y)])

    for id_key, key in enumerate(descriptor_map):

        for id, label in enumerate(y):

            if int(label) in descriptor_map[key]:
                labels_[id] = id_key

    y = labels_

    return ts, X, y
