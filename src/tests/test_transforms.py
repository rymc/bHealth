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
