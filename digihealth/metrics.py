import numpy as np
import pandas as pd
import scipy.stats

class Metrics:

    def __init__(self, timestamps, aggregation_duration, window_overlap):

        # if aggregation_duration == 'day':
        #     duration = 86400
        # elif aggregation_duration == 'hour':
        #     duration = 3600
        # elif aggregation_duration == 'minute':
        #     duration = 60
        # elif aggregation_duration == 'second':
        #     duration = 1

        sampling_frequency = self.establish_sampling_frequency(timestamps)

        indexing = aggregation_duration/sampling_frequency

        if (indexing % 2) != 0:
            indexing = np.ceil(indexing)

        self.window_length = int(indexing)
        self.current_position = 0
        self.window_overlap = window_overlap

    @staticmethod
    def average_labels_per_window(labels, timestamps):
        """Return the average label proportion per window."""
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
        """Return the average duration of a label per window."""
        unique_lab, counts_lab = np.unique(labels, return_counts=True)

        number_of_instances = len(labels)
        number_of_labels = len(unique_lab)
        total_time_in_window = timestamps.iloc[-1] - timestamps.iloc[0]

        label_time_array = np.zeros((number_of_labels, 2))

        for idx, (lab, count) in enumerate(zip(unique_lab, counts_lab)):

            prop = count / number_of_instances
            time_prop = prop * total_time_in_window
            label_time_array[idx, 0] = int(lab)
            label_time_array[idx, 1] = time_prop

        return label_time_array

    @staticmethod
    def number_of_label_changes_per_window(labels, timestamps):
        """Return a confusion matrix of the number of label changes in a window."""
        unique_lab, counts_lab = np.unique(labels, return_counts=True)

        print(unique_lab)

        label_change_array = np.zeros((max(unique_lab)+1, max(unique_lab)+1))

        for idx, lab in labels.iloc[1:].iterrows():
            label_change_array[int(labels.loc[idx-1]), int(labels.loc[idx])] += 1

        return label_change_array

    @staticmethod
    def average_time_between_labels(labels, timestamps, normalise=True):
        """Return the average time, in seconds, between labels in a window."""
        # normalise parameter attempts to remove sequential labels
        # assuming a finite set of ordinal labels
        unique_lab, counts_lab = np.unique(labels, return_counts=True)

        sampling_frequency = 0
        for idx in range(1, timestamps.size):
            sampling_frequency = sampling_frequency + (timestamps.loc[idx].values - timestamps.loc[idx - 1].values)

        sampling_frequency = sampling_frequency / timestamps.size

        sampling_frequency = 1 / sampling_frequency

        number_of_instances = len(labels)
        number_of_labels = len(unique_lab)

        inter_label_times = []

        average_per_label = np.zeros((number_of_labels, 1))

        for idx_outer in range(number_of_labels):
            lab_instance_outer = labels.iloc[idx_outer]
            for idx in range(number_of_instances):
                if (labels.iloc[idx] == lab_instance_outer).any():
                    inter_label_times.append(np.squeeze(timestamps.iloc[idx].values))

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
        """Return the most likely sampling frequency from the timestamps in a time window."""
        sampling_frequency = 0
        for idx in range(1, timestamps.size):
            sampling_frequency = sampling_frequency + (timestamps.loc[idx].values - timestamps.loc[idx-1].values)

        sampling_frequency = sampling_frequency / timestamps.size

        sampling_frequency = 1 / sampling_frequency

        return sampling_frequency

    def slide(self, index, update=True):
        """Slide and return the window of data."""
        window = index[self.current_position - self.window_length:self.current_position]
        if len(window) > 0:
            if len(window.shape) > 1:
                window = window[~np.isnan(window).any(axis=1)]
            else:
                window = window[~np.isnan(window)]
        if update:
            self.current_position += self.window_overlap
        return window

    def localisation_metrics(self, labels, timestamps):
        """Outputs typical localisation metrics."""
        # Room Transfers - Daily average
            # Find all timestamps within a time window
        df_time = pd.DataFrame(timestamps)
        #

        # TODO Number of times bathroom visited during the night
        # TODO Number of times kitchen visited during the night

    def activity_metrics(self, labels, timestamps):
        """Outputs typical activity metrics."""
        # Number of times activities undertaken (e.g. cooking / cleaning) - Daily average
        # Walking - Hourly average
        # Sitting - Hourly average
        # Lying - Hourly average
        # walking - Daily average
        # Main Sleep Length - Daily average
        # Total Sleep Length - Daily average