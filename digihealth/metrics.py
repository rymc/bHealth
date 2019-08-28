"""
metrics.py
====================================
Typical health metrics are contained within this class.
"""

import numpy as np
import pandas as pd

class Metrics:
    """
    Metrics class. This class is used to obtain health metrics from the labels in the data.
    The subjective interpretation of the output is left to the user.

    If you want to add your own metric function, do it here.
    """

    def __init__(self, timestamps, aggregation_duration, window_overlap, fs=None):
        """
        Metrics constructor.

        Parameters
        ----------
        timestamps
           A vector of timestamps.
        aggregation_duration
           Desired aggregation duration.
        window_overlap
           Desired overlap of data windows.
        """

        # if aggregation_duration == 'day':
        #     duration = 86400
        # elif aggregation_duration == 'hour':
        #     duration = 3600
        # elif aggregation_duration == 'minute':
        #     duration = 60
        # elif aggregation_duration == 'second':
        #     duration = 1

        if fs == None:
            sampling_frequency = self.establish_sampling_frequency(timestamps)
        else:
            sampling_frequency = fs

        indexing = aggregation_duration/sampling_frequency

        if (indexing % 2) != 0:
            indexing = np.ceil(indexing)

        self.window_length = int(indexing)
        self.current_position = 0
        self.window_overlap = window_overlap

    @staticmethod
    def average_labels_per_window(labels, timestamps):
        """
        Return the average label proportion per window.

        Parameters
        ----------
        labels
            A vector containing labels.
        timestamps
            A vector containing timestamps.

        Returns
        -------
        label_occurrence_array
            Tuples array, which gives the percentage of occurence of a label in a provided vector.
        """
        unique_lab, counts_lab = np.unique(labels, return_counts=True)

        number_of_instances = len(labels)
        number_of_labels = len(unique_lab)

        label_occurrence_array = np.zeros((number_of_labels, 2))

        for idx, (lab, count) in enumerate(zip(unique_lab, counts_lab)):
            prop = count / number_of_instances
            label_occurrence_array[idx, 0] = int(lab)
            label_occurrence_array[idx, 1] = prop

        return label_occurrence_array

    @staticmethod
    def duration_of_labels_per_window(labels, timestamps):
        """
        Return the average duration of a label per window.

        Parameters
        ----------
        labels
            A vector containing labels.
        timestamps
            A vector containing timestamps.

        Returns
        -------
        label_time_array
            Tuples array, which gives the percentage of time of a given label in a provided vector.
        """
        unique_lab, counts_lab = np.unique(labels, return_counts=True)

        number_of_instances = len(labels)
        number_of_labels = len(unique_lab)
        timestamps = np.array(timestamps)
        total_time_in_window = timestamps[-1] - timestamps[0]

        label_time_array = np.zeros((number_of_labels, 2))

        for idx, (lab, count) in enumerate(zip(unique_lab, counts_lab)):

            prop = count / number_of_instances
            time_prop = prop * total_time_in_window

            # label_time_array.update({lab : float(time_prop)})

            label_time_array[idx, 0] = int(lab)
            label_time_array[idx, 1] = time_prop

        return label_time_array

    @staticmethod
    def number_of_label_changes_per_window(labels, timestamps):
        """
        Return a confusion matrix of the number of label changes in a window.

        Parameters
        ----------
        labels
            A vector containing labels.
        timestamps
            A vector containing timestamps.

        Returns
        -------
        label_change_array
            A NxN array, where N is the number of states, which outlines the number of times each state transitions into another state.
        """
        unique_lab, counts_lab = np.unique(labels, return_counts=True)
        labels_ = np.array(labels)

        label_change_array = np.zeros((len(unique_lab), len(unique_lab)))

        for idx in range(1, len(labels_)):
            label_change_array[int(np.where(unique_lab == labels_[idx-1])[0][0]), int(np.where(unique_lab == labels_[idx])[0][0])] += 1

        return label_change_array

    @staticmethod
    def speed(labels, timestamps, adjacency):
        """
        Return approximate speed of a person given the timestamps and the rate of change of labels, given their distance.

        Parameters
        ----------
        labels
            A vector containing labels.
        timestamps
            A vector containing timestamps.

        Returns
        -------
        speed
            Given the timestamps and adjecency matrix, it outputs likely rate of change of displacement of the labels.
        """
        unique_lab, counts_lab = np.unique(labels, return_counts=True)
        labels_ = np.array(labels)
        adjacency = np.array(adjacency)
        timestamps = np.array(timestamps)

        label_change_array = np.zeros(len(labels))
        label_time_array = np.zeros(len(labels))

        for idx in range(1, len(labels_)):
            label_change_array[idx] += adjacency[int(np.where(unique_lab == labels_[idx-1])[0][0]), int(np.where(unique_lab == labels_[idx])[0][0])]
            label_time_array[idx] += timestamps[idx - 1] - timestamps[idx]

        distances = np.divide(label_change_array, label_time_array)

        return distances

    @staticmethod
    def average_time_between_labels(labels, timestamps, normalise=True):
        """
        Return the average time, in seconds, between labels in a window.

        Parameters
        ----------
        labels
            A vector containing labels.
        timestamps
            A vector containing timestamps.
        normalise
            Flag to normalise the vector, such that two labels next to each other are not considered unique.

        Returns
        -------
        average_per_label
            Given the timestamps and corresponding label vector, returns average time between two unique labels in the array.
        """

        # normalise parameter attempts to remove sequential labels
        # assuming a finite set of ordinal labels
        unique_lab, counts_lab = np.unique(labels, return_counts=True)

        timestamps_ = np.array(timestamps)
        sampling_frequency = 0
        for idx in range(1, len(timestamps_)):
            if (timestamps_[idx] - timestamps_[idx - 1]) < 100:
                sampling_frequency = sampling_frequency + (timestamps_[idx] - timestamps_[idx - 1])

        sampling_frequency = sampling_frequency / len(timestamps_)

        sampling_frequency = 1 / sampling_frequency

        number_of_instances = len(labels)
        labels_ = np.array(labels)
        number_of_labels = len(unique_lab)

        inter_label_times = []

        average_per_label = np.zeros((number_of_labels, 1))

        for idx_outer in range(number_of_labels):
            lab_instance_outer = labels_[idx_outer]
            for idx in range(number_of_instances):
                if (labels_[idx] == lab_instance_outer).any():
                    inter_label_times.append(np.squeeze(timestamps_[idx]))

            inter_label_times = np.diff(inter_label_times)

            if normalise:
                deletion_array = []
                for idx, time in enumerate(inter_label_times):
                    if np.isclose(time, (1/sampling_frequency)):
                        deletion_array.append(idx)
                inter_label_times = np.delete(inter_label_times, deletion_array)

            if inter_label_times.size == 0:
                average_per_label[idx_outer] = 0
            else:
                average_per_label[idx_outer] = np.mean(inter_label_times)
            inter_label_times = []

        return average_per_label

    def establish_sampling_frequency(self, timestamps):
        """
        Return the most likely sampling frequency from the timestamps in a time window. Use only in case actual fs is not available.

        Parameters
        ----------
        timestamps
            A vector containing timestamps.

        Returns
        -------
        sampling_frequency
            Programmatic way of establishing most likely sampling frequency given the timestamps.
        """
        timestamps_ = np.array(timestamps)
        sampling_frequency = 0
        for idx in range(1, len(timestamps_)):
            if (timestamps_[idx] - timestamps_[idx - 1]) < 100:
                sampling_frequency = sampling_frequency + (timestamps_[idx] - timestamps_[idx - 1])

        sampling_frequency = sampling_frequency / len(timestamps_)

        sampling_frequency = 1 / sampling_frequency

        return sampling_frequency

    def slide(self, index, update=True):
        """
        Slide and return the window of data.

        Parameters
        ----------
        index
            Currently holding index of the dataset.
        update
            Update the current position of the window index.

        Returns
        -------
        window
            Windowed data.
        """
        window = index[self.current_position - self.window_length:self.current_position]
        if len(window) > 0:
            if len(window.shape) > 1:
                window = window[~np.isnan(window).any(axis=1)]
            else:
                window = window[~np.isnan(window)]
        if update:
            self.current_position += self.window_overlap
        return window

