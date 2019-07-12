import unittest
import numpy as np
from digihealth.metrics import Metrics

class TestMetrics(unittest.TestCase):

    def test_average_activities_per_window(self):

        winlength = 3

        # two types of label
        labels = [1, 2, 1, 1, 1, 2, 2, 2, 1]

        #monotonically increasing
        timestamp = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        groundtruth = [
                        [[1, 0.67],
                        [2, 0.33]],

                        [[1, 0.67],
                        [2, 0.33]],

                        [[1, 1.]],

                        [[1, 0.67],
                        [2, 0.33]],

                        [[1, 0.33],
                        [2, 0.67]],

                        [[2, 1.]],

                        [[1, 0.33],
                        [2, 0.67]]]

        indicies = np.arange(len(labels))
        metr = Metrics(timestamp, aggregation_duration=3, window_overlap=1)

        metr.current_position = winlength
        final_result = []

        while True:
            windowed_index = metr.slide(np.array(indicies))
            windowed_labels = []
            windowed_time = []

            for win_idx in windowed_index:
                windowed_labels = np.append(windowed_labels, labels[win_idx])
                windowed_time = np.append(windowed_time, timestamp[win_idx])

            if len(windowed_index) > 0 and len(windowed_index) == winlength:
                result = metr.average_labels_per_window(windowed_labels, windowed_time)
                result = np.round(result, 2)
                final_result.append(result)
            else:
                break

        for x, y in zip(final_result, groundtruth):
            np.testing.assert_almost_equal(x, y)

    def test_duration_activities_per_window(self):

        winlength = 3

        # two types of label
        labels = [1, 2, 1, 1, 1, 2, 2, 2, 1]

        #monotonically increasing
        timestamp = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        groundtruth = [
                        [[1, 1.33],
                        [2, 0.67]],

                        [[1, 1.33],
                        [2, 0.67]],

                        [[1, 2.]],

                        [[1, 1.33],
                        [2, 0.67]],

                        [[1, 0.67],
                        [2, 1.33]],

                        [[2, 2.]],

                        [[1, 0.67],
                        [2, 1.33]]]

        indicies = np.arange(len(labels))
        metr = Metrics(timestamp, aggregation_duration=3, window_overlap=1)

        metr.current_position = winlength
        final_result = []

        while True:
            windowed_index = metr.slide(np.array(indicies))
            windowed_labels = []
            windowed_time = []

            for win_idx in windowed_index:
                windowed_labels = np.append(windowed_labels, labels[win_idx])
                windowed_time = np.append(windowed_time, timestamp[win_idx])

            if len(windowed_index) > 0 and len(windowed_index) == winlength:
                result = metr.duration_of_labels_per_window(windowed_labels, windowed_time)
                result = np.round(result, 2)
                final_result.append(result)
            else:
                break

        for x, y in zip(final_result, groundtruth):
            np.testing.assert_almost_equal(x, y)


    def test_average_activity_change_per_window(self):

        winlength = 9

        # two types of label
        labels = [1, 2, 1, 1, 1, 2, 2, 2, 1]

        #monotonically increasing
        timestamp = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        groundtruth = [[ [ 2, 2 ], [ 2, 2 ] ]]

        indicies = np.arange(len(labels))
        metr = Metrics(timestamp, aggregation_duration=9, window_overlap=1)

        metr.current_position = winlength
        final_result = []

        while True:
            windowed_index = metr.slide(np.array(indicies))
            windowed_labels = []
            windowed_time = []

            for win_idx in windowed_index:
                windowed_labels = np.append(windowed_labels, labels[win_idx])
                windowed_time = np.append(windowed_time, timestamp[win_idx])

            if len(windowed_index) > 0 and len(windowed_index) == winlength:
                result = metr.number_of_label_changes_per_window(windowed_labels, windowed_time)
                result = np.round(result, 2)
                final_result.append(result)
            else:
                break

        for x, y in zip(final_result, groundtruth):
            np.testing.assert_almost_equal(x, y)


    def test_average_inter_label_durations(self):

        winlength = 9

        # two types of label
        labels = [1, 2, 1, 1, 1, 2, 2, 2, 1]

        #monotonically increasing
        timestamp = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        groundtruth = [[ [ 3 ], [ 4 ] ]]

        indicies = np.arange(len(labels))
        metr = Metrics(timestamp, aggregation_duration=9, window_overlap=1)

        metr.current_position = winlength
        final_result = []

        while True:
            windowed_index = metr.slide(np.array(indicies))
            windowed_labels = []
            windowed_time = []

            for win_idx in windowed_index:
                windowed_labels = np.append(windowed_labels, labels[win_idx])
                windowed_time = np.append(windowed_time, timestamp[win_idx])

            if len(windowed_index) > 0 and len(windowed_index) == winlength:
                result = metr.average_time_between_labels(windowed_labels, windowed_time, 1)
                result = np.round(result, 2)
                final_result.append(result)
            else:
                break

        for x, y in zip(final_result, groundtruth):
            np.testing.assert_almost_equal(x, y)
