"""
metric_wrappers.py
====================================
The main class containing human-readable wrappers for metrics.
"""

import csv
import numpy as np
import pandas as pd
from bhealth.metrics import Metrics

class Wrapper:
    """
    Wrapper class is instantiated before calling the metrics. The metrics are contained in an array, which is subsequently
    iterated. The outcomes of the iterations are then held inside containers.

    If you want to add your own specific metric function, do it here.
    """

    def __init__(self, labels, timestamps, duration, overlap, fs, label_descriptor_map, csv_prep=None, adjecency=None):
        """
        Wrapper to output the average walking speed.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        duration
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.
        adjecency
            NxN binary matrix of state adjecency, where 1 specifies that two states are adjecent
        label_descriptor_map
            Map of numerical to categorical labels.
        """
        self.labels = labels
        self.timestamps = timestamps
        self.overlap = overlap
        self.fs = fs
        self.csv = csv_prep

        self.adjecency = adjecency
        self.label_descriptor_map = label_descriptor_map

        self.df_time = self.timestamps.astype('datetime64')
        self.df_time = pd.DataFrame(self.df_time, columns=['Time'])
        self.df_label = pd.DataFrame(self.labels, columns=['Label'])

        if duration == 'daily':
            self.duration = 86400
        elif duration == 'hourly':
            self.duration = 3600


    def run_metric_array(self, array):
        """
        Function to run the metric arrays

        Parameters
        ----------
        array
            Vector of metric functions to run.

        Returns
        -------
        metric container
            Container holding the metrics outlined in 'array' parameter. This is indexed by each metric in sequence, followed by the output.
        """

        if self.duration == 3600:
            unique_days = self.df_time['Time'].dt.normalize().unique()
            date_container = []
            metric_container = {}
            for function in array:
                metric_container[function.__name__] = []

            for day in unique_days:
                hour = day

                for hr in range(23):
                    next_hour = hour + np.timedelta64(1, 'h')
                    mask = ((self.df_time['Time'] > hour) & (self.df_time['Time'] <= next_hour))
                    times = self.df_time.loc[mask]
                    labs = self.df_label.loc[mask]

                    if labs.size > 1:

                        date_container.append(hour)

                        times = times.astype(np.int64) // 10 ** 6
                        times = times / 1000

                        metric_holder = []
                        for function in array:
                            metric_holder = (np.apply_along_axis(function, 0, labs, times, self.duration, self.overlap, self.fs).tolist())
                            metric_container[function.__name__].append(metric_holder)

                    hour = next_hour

        elif self.duration == 86400:
            unique_days = self.df_time['Time'].dt.normalize().unique()
            date_container = []
            metric_container = {}
            for function in array:
                metric_container[function.__name__] = []

            for day in unique_days:

                next_day = day + np.timedelta64(1, 'D')
                mask = ((self.df_time['Time'] > day) & (self.df_time['Time'] <= next_day))
                times = self.df_time.loc[mask]
                labs = self.df_label.loc[mask]

                if labs.size > 1:
                    date_container.append(day)

                    times = times.astype(np.int64) // 10 ** 6
                    times = times / 1000

                    metric_holder = []
                    for function in array:
                        metric_holder = (np.apply_along_axis(function, 0, labs, times, self.duration, self.overlap, self.fs).tolist())
                        metric_container[function.__name__].append(metric_holder)

        return metric_container, date_container

    def run_metric_array_csv(self, array):
        """
        Function to run the metric arrays

        Parameters
        ----------
        array
            Vector of metric functions to run.

        Returns
        -------
        metric container
            Container holding the metrics outlined in 'array' parameter. This is indexed by each metric in sequence, followed by the output.
        """

        if self.duration == 3600:
            unique_days = self.df_time['Time'].dt.normalize().unique()
            date_container = []
            metric_container = {}
            metric_container['datetime'] = []
            for function in array:
                metric_container[function.__name__] = []

            for day in unique_days:
                hour = day

                for hr in range(23):
                    next_hour = hour + np.timedelta64(1, 'h')
                    mask = ((self.df_time['Time'] > hour) & (self.df_time['Time'] <= next_hour))
                    times = self.df_time.loc[mask]
                    labs = self.df_label.loc[mask]

                    if labs.size > 1:

                        metric_container['datetime'].append(hour)

                        times = times.astype(np.int64) // 10 ** 6
                        times = times / 1000

                        metric_holder = []
                        for function in array:
                            metric_holder = (np.apply_along_axis(function, 0, labs, times, self.duration, self.overlap, self.fs).tolist())
                            metric_container[function.__name__].append(metric_holder)

                    hour = next_hour

        elif self.duration == 86400:
            unique_days = self.df_time['Time'].dt.normalize().unique()
            date_container = []
            metric_container = {}
            metric_container['datetime'] = []
            for function in array:
                metric_container[function.__name__] = []

            for day in unique_days:

                next_day = day + np.timedelta64(1, 'D')
                mask = ((self.df_time['Time'] > day) & (self.df_time['Time'] <= next_day))
                times = self.df_time.loc[mask]
                labs = self.df_label.loc[mask]

                if labs.size > 1:
                    metric_container['datetime'].append(day)

                    times = times.astype(np.int64) // 10 ** 6
                    times = times / 1000

                    metric_holder = []
                    for function in array:
                        metric_holder = (np.apply_along_axis(function, 0, labs, times, self.duration, self.overlap, self.fs).tolist())
                        metric_container[function.__name__].append(metric_holder)

        csv_out = pd.DataFrame(metric_container)
        csv_out.to_csv(self.csv)

        return metric_container, date_container

    def room_transfers(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the room transfers.

        Parameters
        ----------
        labels
            Vector of labels.
        timestamps
            Vector of timestamps.
        timespan
            Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
            Amount of overlap for the calculation of the transfers.
        fs
            Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
            Output the room transfers.
        """

        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.number_of_label_changes_per_window(labels, timestamps)

        return container

    def duration_walking(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration walking.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration walking.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings(container, 1, 'walking')
        return container

    def duration_sitting(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration sitting.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration sitting.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings(container, 1, 'sitting')
        return container

    def duration_sleeping(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration sleeping.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration sleeping.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings(container, 1, 'sleeping')
        return container

    def duration_washing(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration washing.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration washing.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings(container, 1, 'washing')
        return container

    def duration_eating(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration eating.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration eating.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings(container, 1, 'eating')
        return container

    def duration_studying(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration studying.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration studying.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings(container, 1, 'studying')
        return container

    def duration_in_bathroom(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration studying.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration studying.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings_localisation(container, 'bathroom')
        return container

    def duration_in_bedroom_1(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration studying.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration studying.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings_localisation(container, 'bedroom 1')
        return container

    def duration_in_bedroom_2(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration studying.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration studying.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings_localisation(container, 'bedroom 2')
        return container

    def duration_in_kitchen(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration studying.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration studying.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings_localisation(container, 'kitchen')
        return container

    def duration_in_living_room(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the duration studying.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the duration studying.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings_localisation(container, 'living room')
        return container

    def number_of_bathroom_visits(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the number of bathroom visits.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the number of bathroom visits.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.number_of_label_changes_per_window(labels, timestamps)
        container = self.label_mappings_localisation(container, 'bathroom')
        return container

    def number_of_living_room_visits(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the number of living room visits.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the number of living room visits.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.number_of_label_changes_per_window(labels, timestamps)
        container = self.label_mappings_localisation(container, 'living_room')
        return container

    def number_of_unique_activities(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the number of unique activities per window.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the number of unique activities per window.
        """


        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings(container, 0)
        return container

    def number_of_unique_locations(self, labels, timestamps, timespan, overlap, fs):
        """
        Wrapper to output the number of unique locations per window.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the number of unique locations per window.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.duration_of_labels_per_window(labels, timestamps)
        container = self.label_mappings_localisation(container)
        return container

    def walking_speed(self, labels, timestamps, timespan, overlap, fs):

        """
        Wrapper to output the average walking speed.

        Parameters
        ----------
        labels
           Vector of labels.
        timestamps
           Vector of timestamps.
        timespan
           Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
        overlap
           Amount of overlap for the calculation of the transfers.
        fs
           Sampling frequency.

        Returns
        -------
        metr.number_of_label_changes_per_window(labels, timestamps)
           Output the average walking speed.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        container = metr.speed(labels, timestamps, self.adjecency)
        return np.abs(container), np.nanmean(np.abs(container)), np.nanmax(np.abs(container))

    # def time_between_upstairs_downstairs(self, labels, timestamps, timespan, overlap, fs=None):
    #     """
    #     Wrapper to output the time taken to walking up to downstairs.
    #
    #     Parameters
    #     ----------
    #     labels
    #        Vector of labels.
    #     timestamps
    #        Vector of timestamps.
    #     timespan
    #        Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
    #     overlap
    #        Amount of overlap for the calculation of the transfers.
    #     fs
    #        Sampling frequency.
    #
    #     Returns
    #     -------
    #     metr.number_of_label_changes_per_window(labels, timestamps)
    #        Output the time taken to walking up to downstairs.
    #     """
    #     metr = Metrics(timestamps, timespan, overlap, fs)
    #     return metr.average_time_between_labels(labels, timestamps)
    #
    # def time_between_downstairs_upstairs(self, labels, timestamps, timespan, overlap, fs=None):
    #     """
    #     Wrapper to output the time taken to walking down to upstairs.
    #
    #     Parameters
    #     ----------
    #     labels
    #        Vector of labels.
    #     timestamps
    #        Vector of timestamps.
    #     timespan
    #        Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
    #     overlap
    #        Amount of overlap for the calculation of the transfers.
    #     fs
    #        Sampling frequency.
    #
    #     Returns
    #     -------
    #     metr.number_of_label_changes_per_window(labels, timestamps)
    #        Output the time taken to walking down to upstairs.
    #     """
    #     metr = Metrics(timestamps, timespan, overlap, fs)
    #     return metr.average_time_between_labels(labels, timestamps)
    #
    # def sit_to_stand_transitions(self, labels, timestamps, timespan, overlap, fs=None):
    #     """
    #     Wrapper to output the sit to stand transitions.
    #
    #     Parameters
    #     ----------
    #     labels
    #         Vector of labels.
    #     timestamps
    #         Vector of timestamps.
    #     timespan
    #         Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
    #     overlap
    #         Amount of overlap for the calculation of the transfers.
    #     fs
    #         Sampling frequency.
    #
    #     Returns
    #     -------
    #     metr.number_of_label_changes_per_window(labels, timestamps)
    #         Output the sit to stand transitions.
    #     """
    #     metr = Metrics(timestamps, timespan, overlap, fs)
    #     container = metr.number_of_label_changes_per_window(labels, timestamps)
    #     container = self.label_mappings(container, 1)
    #
    #     return container
    #
    # def sleep_efficiency(self, labels, timestamps, timespan, overlap, fs=None):
    #     """
    #     Wrapper to output the sleep efficiency.
    #
    #     Parameters
    #     ----------
    #     labels
    #         Vector of labels.
    #     timestamps
    #         Vector of timestamps.
    #     timespan
    #         Duration of metric calculation. Either 86400 (daily) or 3600 (hourly)
    #     overlap
    #         Amount of overlap for the calculation of the transfers.
    #     fs
    #         Sampling frequency.
    #
    #     Returns
    #     -------
    #     metr.number_of_label_changes_per_window(labels, timestamps)
    #         Output the sleep efficiency.
    #     """
    #
    #     metr = Metrics(timestamps, timespan, overlap, fs)
    #     container = metr.number_of_label_changes_per_window(labels, timestamps)
    #     return container

    def label_reduction(self, labels):

        labels_ = np.zeros([len(labels)])

        for id_key, key in enumerate(self.label_descriptor_map):

            for id, label in enumerate(labels):

                if int(label) in self.label_descriptor_map[key]:
                    labels_[id] = id_key

        return labels_

    def label_mappings_localisation(self, container, label_to_extract=None):

        container_ = []
        returner_ = {}

        if label_to_extract is None:
            for key in self.label_descriptor_map:
                for id, label in enumerate(container):
                    if int(label[0]) in self.label_descriptor_map[key]:
                        if self.csv is not None:
                            returner_.append(label[1])
                        else:
                            returner_.update({key : label[1]})
        else:
            for id, label in enumerate(container):
                if int(label[0]) in self.label_descriptor_map[label_to_extract]:
                    container_.append(label[1])

            container_ = np.sum(container_)
            if self.csv is not None:
                returner_ = container_
            else:
                returner_ = {label_to_extract : container_}

        return returner_

    def label_mappings(self, container, is_duration, label_to_extract=None):

        # if house_ == 'A' or house_ == 0:
        #
        #     descriptor_map = {
        #         77: 'cooking',
        #         78: 'sitting',
        #         79: 'sitting',
        #         80: 'washing',
        #         81: 'sleeping',
        #         82: 'sitting'}
        #
        # elif house_ == 'B' or house_ == 1:
        #
        #     descriptor_map = {
        #         77: 'cooking',
        #         78: 'washing',
        #         79: 'sitting',
        #         80: 'sitting',
        #         81: 'sleeping',
        #         82: 'sleeping'}
        #
        # elif house_ == 'C' or house_ == 2:
        #
        #     descriptor_map = {
        #         77: 'sitting',
        #         78: 'cooking',
        #         79: 'washing',
        #         80: 'sitting',
        #         81: 'sleeping',
        #         82: 'sitting'}
        #
        # elif house_ == 'D' or house_ == 3:
        #
        #     descriptor_map = {
        #         77: 'washing',
        #         78: 'sleeping',
        #         79: 'sitting',
        #         80: 'sitting',
        #         81: 'sleeping',
        #         82: 'cooking'}
        container_ = []

        if self.csv is not None:
            returner_ = []
        else:
            returner_ = {}

        if label_to_extract is None:
            for key in self.label_descriptor_map:
                for id, label in enumerate(container):
                    if int(label[0]) == self.label_descriptor_map[key]:
                        if self.csv is not None:
                            returner_.append(label[1])
                        else:
                            returner_.update({key : label[1]})
        else:
            for id, label in enumerate(container):
                if int(label[0]) == self.label_descriptor_map[label_to_extract]:
                    container_.append(label[1])

            if is_duration:
                container_ = np.sum(container_)
                if self.csv is not None:
                    returner_ = container_
                else:
                    returner_ = {label_to_extract : container_}

        return returner_
