import unittest
import numpy as np
from src.transforms import Transforms

class TestTransforms(unittest.TestCase):
    def test_mean_crossings(self):
        x_list = [[-1, -1, -1],
                  [0, 0, 0],
                  [1, 5, 9],
                  [-1, 1, -2, 2, -3, 3]]
        y_list = [0, 0, 1, 5]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.mean_crossings(x)
            self.assertEqual(result, y)

        winlength = 3
        x_list = [[-1, -1, -1, -1, -1, -1],
                  [0, 0, 0, 0, 0, 0],
                  [1, -5, 2, -6, 9, -3, 2, -5, 2],
                  [-1, 1, 2, 2, -3, 3]]
        y_list = [[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [2, 2, 2, 2, 1, 2, 2],
                  [1, 1, 1, 2]]
        t = Transforms(window_length=winlength, window_overlap=1)

        for x, y in zip(x_list, y_list):
            t.current_position = winlength
            final_result = []
            while True:
                windowed_raw = t.slide(np.array(x))
                if len(windowed_raw) > 0 and len(windowed_raw) == winlength:
                    result = t.mean_crossings(windowed_raw)
                    final_result.append(result)
                else:
                    break
            self.assertEqual(final_result, y)

    def test_zero_crossings(self):
        x_list = [[-1, -1, -1],
                  [0, 0, 0],
                  [1, -4, 5, -10],
                  [2, -3, 4, -5]]
        y_list = [0, 0, 3, 3]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.zero_crossings(x)
            self.assertEqual(result, y)

        winlength = 3
        x_list = [[-1, -1, -1, -1, -1, -1],
                  [0, 0, 0, 0, 0, 0],
                  [1, -5, 2, -6, 9, -3, 2, -5, 2],
                  [-1, 1, 2, 2, -3, 3]]
        y_list = [[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [2, 2, 2, 2, 2, 2, 2],
                  [1, 0, 1, 2]]
        t = Transforms(window_length=winlength, window_overlap=1)

        for x, y in zip(x_list, y_list):
            t.current_position = winlength
            final_result = []
            while True:
                windowed_raw = t.slide(np.array(x))
                if len(windowed_raw) > 0 and len(windowed_raw) == winlength:
                    result = t.zero_crossings(windowed_raw)
                    final_result.append(result)
                else:
                    break
            self.assertEqual(final_result, y)

    def test_25_percentiles(self):
        x_list = [[1, 2, 3, 4, 5, 6, 7],
                  [5, 6, 3, 1, 3],
                  [10, -10, -90, -90.5, -100, -200]]
        y_list = [2.5, 3, -97.625]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.p25(x)
            self.assertEqual(result, y)

        winlength = 3
        x_list = [[1, 2, 3, 4, 5, 6, 7],
                  [5, 6, 3, 1, 3],
                  [10, -10, -90, -90.5, -100, -200]]
        y_list = [[1.5, 2.5, 3.5, 4.5, 5.5],
                  [4, 2, 2],
                  [-50, -90.25, -95.25, -150]]
        t = Transforms(window_length=winlength, window_overlap=1)

        for x, y in zip(x_list, y_list):
            t.current_position = winlength
            final_result = []
            while True:
                windowed_raw = t.slide(np.array(x))
                if len(windowed_raw) > 0 and len(windowed_raw) == winlength:
                    result = t.p25(windowed_raw)
                    final_result.append(result)
                else:
                    break
            np.testing.assert_almost_equal(final_result, y)

    def test_75_percentiles(self):
        x_list = [[5, 6, 7, 8, 9],
                  [5,5.5,4.5,4.75,5.35,5.45,6.15],
                  [-10, -40, 50, 2, 4, 5]]
        y_list = [8, 5.475, 4.75]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.p75(x)
            self.assertEqual(result, y)

        winlength = 3
        x_list = [[1, 2, 3, 4, 5, 6, 7],
                  [5, 6, 3, 1, 3],
                  [10, -10, -90, -90.5, -100, -200]]
        y_list = [[2.5, 3.5, 4.5, 5.5, 6.5],
                  [5.5, 4.5, 3],
                  [0, -50, -90.25, -95.25]]
        t = Transforms(window_length=winlength, window_overlap=1)

        for x, y in zip(x_list, y_list):
            t.current_position = winlength
            final_result = []
            while True:
                windowed_raw = t.slide(np.array(x))
                if len(windowed_raw) > 0 and len(windowed_raw) == winlength:
                    result = t.p75(windowed_raw)
                    final_result.append(result)
                else:
                    break
            np.testing.assert_almost_equal(final_result, y)

    def test_interquartiles(self):
        x_list = [[5, 6, 7, 8, 9, 10],
                  [1,2,3,4,2,3,4,10,5,7,8,11],
                  [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
        y_list = [2.5, 4.5, 88]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.interq(x)
            self.assertEqual(result, y)

        winlength = 3
        x_list = [[1, 2, 3, 4, 5, 6, 7],
                  [5, 6, 3, 1, 3],
                  [10, -90, -10, -90.5, -100, -200]]
        y_list = [[1, 1, 1, 1, 1],
                  [1.5, 2.5, 1],
                  [50, 40.25, 45, 54.75]]
        t = Transforms(window_length=winlength, window_overlap=1)
        for x, y in zip(x_list, y_list):
            t.current_position = winlength
            final_result = []
            while True:
                windowed_raw = t.slide(np.array(x))
                if len(windowed_raw) > 0 and len(windowed_raw) == winlength:
                    result = t.interq(windowed_raw)
                    final_result.append(result)
                else:
                    break
            np.testing.assert_almost_equal(final_result, y)

    def test_skewness(self):
            x_list = [[5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 2, 3, 4, 10, 5, 7, 8, 11],
                      [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
            y_list = [0, 0.6486028516711329, 0.05039822936776113]
            t = Transforms(window_length=1, window_overlap=0)
            for x, y in zip(x_list, y_list):
                result = t.skewn(x)
                np.testing.assert_almost_equal(result, y)

            winlength = 3
            x_list = [[1, 2, 3, 4, 5, 6, 7],
                      [5, 6, 3, 1, 3],
                      [10, -90, -10, -90.5, -100, -200]]
            y_list = [[0, 0, 0, 0, 0],
                      [-0.3818018, 0.2390631, -0.7071068],
                      [-0.5951701, 0.7070141, 0.6778576, -0.687648]]
            t = Transforms(window_length=winlength, window_overlap=1)
            for x, y in zip(x_list, y_list):
                t.current_position = winlength
                final_result = []
                while True:
                    windowed_raw = t.slide(np.array(x))
                    if len(windowed_raw) > 0 and len(windowed_raw) == winlength:
                        result = t.skewn(windowed_raw)
                        final_result.append(result)
                    else:
                        break
                np.testing.assert_almost_equal(final_result, y)

    def test_kurtosis(self):
        x_list = [[5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 2, 3, 4, 10, 5, 7, 8, 11],
                  [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
        y_list = [1.7314285714285718, 2.128698649813272, 1.8298588697064042]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.kurtosis(x)
            self.assertEqual(result, y)

        winlength = 5
        x_list = [[1, 2, 3, 4, 5, 6, 7],
                  [5, 6, 3, 1, 3],
                  [10, -90, -10, -90.5, -100, -200]]
        y_list = [[1.7, 1.7, 1.7],
                  [1.7957064],
                  [1.2987871, 2.5169668]]
        t = Transforms(window_length=winlength, window_overlap=1)
        for x, y in zip(x_list, y_list):
            t.current_position = winlength
            final_result = []
            while True:
                windowed_raw = t.slide(np.array(x))
                if len(windowed_raw) > 0 and len(windowed_raw) == winlength:
                    result = t.kurtosis(windowed_raw)
                    final_result.append(result)
                else:
                    break
            np.testing.assert_almost_equal(final_result, y)

    def test_spec_energy(self):
        x_list = [[5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 2, 3, 4, 10, 5, 7, 8, 11],
                  [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
        y_list = [2130, 5016, 1661139.9999999995]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.spec_energy(x)
            np.testing.assert_almost_equal(result, y)

    def test_spec_entropy(self):
        x_list = [[5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 2, 3, 4, 10, 5, 7, 8, 11],
                  [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
        y_list = [0.9333118993685063, 1.8492364375192922, 1.941214222324683]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.spec_entropy(x)
            np.testing.assert_almost_equal(result, y)