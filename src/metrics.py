import numpy as np
import scipy.stats

class Metrics:

    def __init__(self, timestamps, aggregation_duration, window_overlap):

        sampling_frequency = self.establish_sampling_frequency(timestamps)

        indexing = aggregation_duration/sampling_frequency

        if (indexing % 2) != 0:
            indexing = np.ceil(indexing)

        self.window_length = int(indexing)
        self.current_position = 0
        self.window_overlap = window_overlap

    @staticmethod
    def average_labels_per_window(labels, timestamps):
        # assuming a finite set of ordinal labels
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
        # assuming a finite set of ordinal labels
        unique_lab, counts_lab = np.unique(labels, return_counts=True)

        number_of_instances = len(labels)
        number_of_labels = len(unique_lab)
        total_time_in_window = timestamps[-1] - timestamps[0]

        label_time_array = np.zeros((number_of_labels, 2))

        for idx, (lab, count) in enumerate(zip(unique_lab, counts_lab)):

            prop = count / number_of_instances
            time_prop = prop * total_time_in_window
            label_time_array[idx, 0] = int(lab)
            label_time_array[idx, 1] = time_prop

        return label_time_array

    @staticmethod
    def number_of_label_changes_per_window(labels, timestamps):
        # assuming a finite set of ordinal labels
        unique_lab, counts_lab = np.unique(labels, return_counts=True)

        number_of_instances = len(labels)
        number_of_labels = len(unique_lab)
        total_time_in_window = timestamps[-1] - timestamps[0]

        label_change_array = np.zeros((number_of_labels, number_of_labels))

        for idx_outer in range(number_of_labels):
            lab_instance_outer = labels[idx_outer]
            for idx_inner in range(number_of_labels):
                lab_instance_inner = labels[idx_inner]
                for idx in range(number_of_instances - 1):

                    if labels[idx] == lab_instance_outer and labels[idx+1] == lab_instance_inner:
                        label_change_array[int(idx_outer), int(idx_inner)] += 1

        return label_change_array

    # @staticmethod
    # def average_time_between_labels(labels, timestamps):
    #     # assuming a finite set of ordinal labels
    #     unique_lab, counts_lab = np.unique(labels, return_counts=True)
    #
    #     number_of_instances = len(labels)
    #     number_of_labels = len(unique_lab)
    #     total_time_in_window = timestamps[-1] - timestamps[0]
    #
    #     label_change_array = np.zeros((number_of_labels, number_of_labels))
    #
    #     for idx_outer in range(number_of_labels):
    #         lab_instance_outer = labels[idx_outer]
    #         if labels[idx] == lab_instance_outer and labels[idx + 1] == lab_instance_inner:
    #             label_change_array[int(idx_outer), int(idx_inner)] += 1
    #
    #     return label_change_array

    def establish_sampling_frequency(self, timestamps):
        sampling_frequency = []
        for idx, time in enumerate(timestamps):
            if idx >= 1:
                sampling_frequency.append(time - previous_time)
            previous_time = time

        sampling_frequency = 1 / np.mean(sampling_frequency)

        return sampling_frequency

    def slide(self, index, update=True):
        window = index[self.current_position - self.window_length:self.current_position]
        if len(window) > 0:
            if len(window.shape) > 1:
                window = window[~np.isnan(window).any(axis=1)]
            else:
                window = window[~np.isnan(window)]
        if update:
            self.current_position += self.window_overlap
        return window
