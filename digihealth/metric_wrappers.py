import numpy as np
import pandas as pd
from digihealth.metrics import Metrics

class Wrapper:

    def __init__(self, labels, timestamps, duration, overlap, fs, adjecency, label_descriptor_map):
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
            for day in unique_days:
                hour = day

                metric_container = {}
                for function in array:
                    metric_container[function.__name__] = []

                for hr in range(23):
                    hour_container = {}
                    next_hour = hour + np.timedelta64(1, 'h')
                    mask = ((self.df_time['Time'] > hour) & (self.df_time['Time'] <= next_hour))
                    times = self.df_time.loc[mask]
                    labs = self.df_label.loc[mask]

                    if labs.size > 1:

                        times = times.astype(np.int64) // 10 ** 6
                        times = times / 1000

                        metric_holder = []
                        for function in array:
                            metric_holder.extend((np.apply_along_axis(function, 0, labs, times, self.duration, self.overlap).tolist()))
                            metric_container[function.__name__].append(metric_holder)

                    hour = next_hour

        elif self.duration == 86400:
            unique_days = self.df_time['Time'].dt.normalize().unique()
            for day in unique_days:

                metric_container = {}
                for function in array:
                    metric_container[function.__name__] = []

                next_day = day + np.timedelta64(1, 'D')
                mask = ((self.df_time['Time'] > day) & (self.df_time['Time'] <= next_day))
                times = self.df_time.loc[mask]
                labs = self.df_label.loc[mask]

                if labs.size > 1:

                    times = times.astype(np.int64) // 10 ** 6
                    times = times / 1000

                    metric_holder = []
                    for function in array:
                        metric_holder.extend(
                            (np.apply_along_axis(function, 0, labs, times, self.duration, self.overlap).tolist()))
                        metric_container[function.__name__].append(metric_holder)

        return metric_container

    def room_transfers(self, labels, timestamps, timespan, overlap, fs=None):
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
        return metr.number_of_label_changes_per_window(labels, timestamps)

    def sit_to_stand_transitions(self, labels, timestamps, timespan, overlap, fs=None):
        """
        Wrapper to output the sit to stand transitions.

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
            Output the sit to stand transitions.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.number_of_label_changes_per_window(labels, timestamps)

    def sleep_efficiency(self, labels, timestamps, timespan, overlap, fs=None):
        """
        Wrapper to output the sleep efficiency.

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
            Output the sleep efficiency.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.number_of_label_changes_per_window(labels, timestamps)

    def sleep_quality(self, labels, timestamps, timespan, overlap, fs=None):
        """
        Wrapper to output the sleep quality.

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
            Output the sleep quality.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.number_of_label_changes_per_window(labels, timestamps)

    def duration_walking(self, labels, timestamps, timespan, overlap, fs=None):
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
        return metr.duration_of_labels_per_window(labels, timestamps)

    def duration_sitting(self, labels, timestamps, timespan, overlap, fs=None):
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
        return metr.duration_of_labels_per_window(labels, timestamps)

    def duration_lying(self, labels, timestamps, timespan, overlap, fs=None):
        """
        Wrapper to output the duration lying.

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
           Output the duration lying.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.duration_of_labels_per_window(labels, timestamps)

    def average_duration_walking(self, labels, timestamps, timespan, overlap, fs=None):
        """
        Wrapper to output the average duration walking.

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
           Output the average duration walking.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.duration_of_labels_per_window(labels, timestamps)

    def number_of_bathroom_visits(self, labels, timestamps, timespan, overlap, fs=None):
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
        return metr.average_labels_per_window(labels, timestamps)

    def number_of_kitchen_visits(self, labels, timestamps, timespan, overlap, fs=None):
        """
        Wrapper to output the number of kitchen visits.

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
           Output the number of kitchen visits.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.average_labels_per_window(labels, timestamps)

    def number_of_unique_activities(self, labels, timestamps, timespan, overlap, fs=None):
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
        return metr.average_labels_per_window(labels, timestamps)

    def time_between_upstairs_downstairs(self, labels, timestamps, timespan, overlap, fs=None):
        """
        Wrapper to output the time taken to walking up to downstairs.

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
           Output the time taken to walking up to downstairs.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.average_time_between_labels(labels, timestamps)

    def time_between_downstairs_upstairs(self, labels, timestamps, timespan, overlap, fs=None):
        """
        Wrapper to output the time taken to walking down to upstairs.

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
           Output the time taken to walking down to upstairs.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.average_time_between_labels(labels, timestamps)

    def average_walking_speed(self, labels, timestamps, timespan, overlap, fs=None):
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
        return np.mean(metr.speed(labels, timestamps, self.adjecency))

    def max_walking_speed(self, labels, timestamps, timespan, overlap, fs=None):
        """
        Wrapper to output the max walking speed.

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
           Output the max walking speed.
        """
        metr = Metrics(timestamps, timespan, overlap, fs)
        return np.max(metr.speed(labels, timestamps, self.adjecency))