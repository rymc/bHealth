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

    def test_25_percentiles(self):
        x_list = [[1, 2, 3, 4, 5, 6, 7],
                  [5, 6, 3, 1, 3],
                  [10, -10, -90, -90.5, -100, -200]]
        y_list = [2.5, 3, -97.625]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.p25(x)
            self.assertEqual(result, y)

    def test_75_percentiles(self):
        x_list = [[5, 6, 7, 8, 9],
                  [5,5.5,4.5,4.75,5.35,5.45,6.15],
                  [-10, -40, 50, 2, 4, 5]]
        y_list = [8, 5.475, 4.75]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.p75(x)
            self.assertEqual(result, y)

    def test_interquartiles(self):
        x_list = [[5, 6, 7, 8, 9, 10],
                  [1,2,3,4,2,3,4,10,5,7,8,11],
                  [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
        y_list = [2.5, 4.5, 88]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.interq(x)
            self.assertEqual(result, y)

    def test_skewness(self):
            x_list = [[5, 6, 7, 8, 9, 10],
                      [1, 2, 3, 4, 2, 3, 4, 10, 5, 7, 8, 11],
                      [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
            y_list = [0, 0.6486028516711329, 0.05039822936776113]
            t = Transforms(window_length=1, window_overlap=0)
            for x, y in zip(x_list, y_list):
                result = t.skewn(x)
                self.assertEqual(result, y)

    def test_kurtosis(self):
        x_list = [[5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 2, 3, 4, 10, 5, 7, 8, 11],
                  [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
        y_list = [-1.2685714285714282, -0.8713013501867279, -1.1701411302935958]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.kurtosis(x)
            self.assertEqual(result, y)

    def test_spec_energy(self):
        x_list = [[5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 2, 3, 4, 10, 5, 7, 8, 11],
                  [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
        y_list = [2130, 5016, 1661139.9999999995]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.spec_energy(x)
            self.assertEqual(result, y)

    def test_spec_entropy(self):
        x_list = [[5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 2, 3, 4, 10, 5, 7, 8, 11],
                  [7, 7, 31, 31, 47, 75, 87, 115, 116, 119, 119, 155, 177]]
        y_list = [0.9333118993685063, 1.8492364375192922, 1.941214222324683]
        t = Transforms(window_length=1, window_overlap=0)
        for x, y in zip(x_list, y_list):
            result = t.spec_entropy(x)
            self.assertEqual(result, y)