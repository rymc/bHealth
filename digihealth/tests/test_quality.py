"""
test_quality.py
====================================
Unit tests for DataQuality class.
"""

import unittest
import numpy as np
import pandas as pd
import random
from numpy.random import randn
from digihealth.data_quality import DataQuality
from scipy import stats

class TestQuality(unittest.TestCase):
    """
    Unit test class for DataQuality.

    If you want to add your own quality function, do it here.
    """

    def test_continuity(self):
        """
        Test continuity.
        """

        quali = DataQuality('H')
        #Generate some timestamps
        timestamps_nominal = pd.date_range(start='6/1/2019', end='6/2/2019', freq='H')
        number_of_timestamps = len(timestamps_nominal)

        #Randomly drop 70% of the stamps
        indecies_to_retain = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23]
        timestamps_to_check = timestamps_nominal[indecies_to_retain]

        #Capture groundtruth
        groundtruth = [0.125]

        #Check for continuity
        reduction = quali.check_continuity(timestamps_to_check)
        #Compare
        np.testing.assert_almost_equal(reduction, groundtruth)

    def test_variance(self):
        """
        Test variance.
        """

        number_of_columns = 5
        number_of_rows = 200

        data_ = np.zeros([number_of_rows, number_of_columns])

        quali = DataQuality('H')

        for col in range(1, number_of_columns):
            for row in range(0, number_of_rows):
                data_[row, col] = random.randint(1,10)

        alert_, variance_ = quali.check_variance(data_)

        np.testing.assert_almost_equal(alert_, 1)

    def test_anomalies(self):
        """
        Test anomalies.
        """

        number_of_columns = 5
        number_of_rows = 200

        data_ = np.zeros([number_of_rows, number_of_columns])

        quali = DataQuality('H')

        for col in range(0, number_of_columns):
            for row in range(0, number_of_rows):
                data_[row, col] = random.randint(1, 10)

        data_[57, 1] = 1000
        data_[100, 4] = 1000

        groundtruth = np.zeros((number_of_rows, number_of_columns))
        groundtruth[57, 1] = 1
        groundtruth[100, 4] = 1

        outliers_ = quali.check_anomalies(data_)

        np.testing.assert_almost_equal(outliers_, groundtruth)

    def test_correlations(self):
        """
        Test contents.
        """

        number_of_columns = 2
        number_of_rows = 200

        data_ = np.zeros([number_of_rows, number_of_columns])

        quali = DataQuality('H')

        for col in range(0, number_of_columns):
            for row in range(0, number_of_rows):
                data_[row, col] = random.randint(1, 2)

        covariances_gt = np.cov(data_.T)
        pearson_gt = np.zeros((number_of_columns, number_of_columns))
        spearman_gt = np.zeros((number_of_columns, number_of_columns))

        for outer_ in range(0, number_of_columns):
            for inner_ in range(0, number_of_columns):
                # check Pearson's
                pearson_gt[outer_, inner_], _ = stats.pearsonr(data_[:, outer_], data_[:, inner_])

                # check Spearman's
                spearman_gt[outer_, inner_], _ = stats.spearmanr(data_[:, outer_], data_[:, inner_])


        covariances_, pearson_, spearman_ = quali.check_correlations(data_)

        np.testing.assert_almost_equal(covariances_, covariances_gt)
        np.testing.assert_almost_equal(pearson_, pearson_gt)
        np.testing.assert_almost_equal(spearman_, spearman_gt)

    def test_uniqueness(self):

        quali = DataQuality('H')
        # Generate some timestamps
        timestamps_nominal = pd.Series(['3/11/2000', '3/11/2000', '3/13/2000'])
        timestamps_nominal = pd.to_datetime(timestamps_nominal)

        groundtruth = [0.6666666667]

        # Check for continuity
        uniqueness = quali.check_uniqueness(timestamps_nominal)

        np.testing.assert_almost_equal(uniqueness, groundtruth)