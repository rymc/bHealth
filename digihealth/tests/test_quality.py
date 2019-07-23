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
        indecies_to_retain = [3, 4, 5, 6, 9, 12, 13, 20]
        timestamps_to_check = timestamps_nominal[indecies_to_retain]

        #Capture groundtruth
        groundtruth = [0.5555556]

        #Check for continuity
        final_result_cont, final_result_reduction = quali.check_continuity(timestamps_to_check)
        #Compare
        np.testing.assert_almost_equal(final_result_reduction, groundtruth)
