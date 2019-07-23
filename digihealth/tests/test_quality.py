import unittest
import numpy as np
import pandas as pd
import random
from digihealth.data_quality import DataQuality

class TestQuality(unittest.TestCase):

    def test_continuity(self):

        quali = DataQuality(0, 2, 3, 'H')
        #Generate some timestamps
        timestamps_nominal = pd.date_range(start='6/1/2019', end='6/2/2019', freq='H')
        number_of_timestamps = len(timestamps_nominal)

        #Randomly drop 70% of the stamps
        indecies_to_retain = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23]
        timestamps_to_check = timestamps_nominal[indecies_to_retain]

        #Capture groundtruth
        groundtruth = [0.125]

        #Check for continuity
        alert_, reduction = quali.check_continuity(timestamps_to_check)
        #Compare
        np.testing.assert_almost_equal(alert_, 1)

    def test_variance(self):

        number_of_columns = 5
        number_of_rows = 200

        data_ = np.zeros([number_of_rows, number_of_columns])

        quali = DataQuality(0, 2, 3, 'H')

        for col in range(1, number_of_columns):
            for row in range(0, number_of_rows):
                data_[row, col] = random.randint(1,10)

        alert_, variance_ = quali.check_variance(data_)

        np.testing.assert_almost_equal(alert_, 1)

    def test_anomalies(self):

        number_of_columns = 5
        number_of_rows = 200

        data_ = np.zeros([number_of_rows, number_of_columns])

        quali = DataQuality(0, 2, 3, 'H')

        for col in range(0, number_of_columns):
            for row in range(0, number_of_rows):
                data_[row, col] = random.randint(1, 10)

        data_[57, 1] = 1000
        data_[100, 4] = 1000

        alert_, outliers_ = quali.check_anomalies(data_)

        np.testing.assert_almost_equal(alert_, 1)