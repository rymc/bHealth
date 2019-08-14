import numpy as np
import pandas as pd
from digihealth.metrics import Metrics

class Wrapper:

    def __init__(self, labels, timestamps, duration, overlap, fs=None):
        self.labels = labels
        self.timestamps = timestamps
        self.overlap = overlap
        self.fs = fs

        self.df_time = self.timestamps.astype('datetime64')
        self.df_time = pd.DataFrame(self.df_time, columns=['Time'])
        self.df_label = pd.DataFrame(self.labels, columns=['Label'])

        if duration == 'daily':
            self.duration = 86400
        elif duration == 'hourly':
            self.duration = 3600

    def run_metric_array(self, array):


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
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.number_of_label_changes_per_window(labels, timestamps)

    def sit_to_stand_transitions(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.number_of_label_changes_per_window(labels, timestamps)

    def sleep_efficiency(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.number_of_label_changes_per_window(labels, timestamps)

    def sleep_quality(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.number_of_label_changes_per_window(labels, timestamps)

    def duration_walking(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.duration_of_labels_per_window(labels, timestamps)

    def duration_sitting(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.duration_of_labels_per_window(labels, timestamps)

    def duration_lying(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.duration_of_labels_per_window(labels, timestamps)

    def average_duration_walking(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.duration_of_labels_per_window(labels, timestamps)

    def number_of_bathroom_visits(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.average_labels_per_window(labels, timestamps)

    def number_of_kitchen_visits(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.average_labels_per_window(labels, timestamps)

    def number_of_unique_activities(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.average_labels_per_window(labels, timestamps)

    def time_between_upstairs_downstairs(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.average_time_between_labels(labels, timestamps)

    def time_between_downstairs_upstairs(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return metr.average_time_between_labels(labels, timestamps)

    def average_walking_speed(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return np.mean(metr.speed(labels, timestamps))

    def max_walking_speed(self, labels, timestamps, timespan, overlap, fs=None):
        metr = Metrics(timestamps, timespan, overlap, fs)
        return np.max(metr.speed(labels, timestamps))