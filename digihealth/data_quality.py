"""
data_quality.py
====================================
Data quality methods are contained within this class.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class DataQuality:
    """
    Data Quality class is used to give the users some information about your dataset.
    It can help give the user basic information about the quality of the set. If there are any discrepancies
    it will raise an 'alert' and print a string.

    If you want to add your own quality function, do it here.
    """

    def __init__(self, sample_rate):
        """
        DataQuality constructor.

        Parameters
        ----------
        sample_rate
            Sample rate of the provided data.
        """

        self.sample_rate = sample_rate

        # # Continuity
        # self.continuity = self.check_continuity(timestamps)
        #
        # # Variance
        # self.variance = self.check_variance(dataset)
        #
        # # Anomalies
        # self.anomalies = self.check_anomalies(dataset)
        #
        # # Correlations
        # self.covariances_, self.pearson_, self.spearman_ = self.check_correlations(dataset)

    def evaluate_point(self, point, means, covariances):
        """
        Evaluate the probability of a data point given multivariate mean and covariance matrices.

        Parameters
        ----------
        point
            A point to evaluate.
        means
            A vector of means to evaluate.
        covariances
            A matrix of covariances.

        Returns
        ----------
        prob
            Probability of a point being part of a distribtion
        """
        m_dist_x = np.dot((point - means).transpose(), np.linalg.inv(covariances))
        m_dist_x = np.dot(m_dist_x, (point - means))
        prob = 1 - stats.chi2.cdf(m_dist_x, 3)
        return prob

    def check_continuity(self, timestamps):
        """
        Check the continuity of the dataset.

        Parameters
        ----------
        timestamps
            A vector of timestamps. As default, this can be as numpy time array.

        Returns
        ----------
        reduction_
            Proportion of reduction of the entire dataset, given the discontinous set of data.
        """
        df_time = timestamps.values.astype('datetime64')
        number_of_instances_prior = len(df_time)
        x = np.ones(number_of_instances_prior)
        df_time = pd.DataFrame({'Time': df_time})

        possible_times = pd.date_range(start=pd.to_datetime(df_time['Time'][0]), end=pd.to_datetime(df_time['Time'][number_of_instances_prior-1]), freq=self.sample_rate).time
        df_time = df_time.set_index('Time').reindex(possible_times).reset_index().reindex(columns=df_time.columns)
        df_time = df_time.ffill()

        number_of_instances_post = len(df_time)
        reduction_ = (number_of_instances_post - number_of_instances_prior)/number_of_instances_post

        if reduction_ == 0:
            alert_ = 0
        else:
            alert_ = 1
            self.alert_raised_ = 1
            red = reduction_ * 100
            print('The set appears discontinuous. From what I can see, there is almost', red, 'percent missing. Check for completeness of the data.')

        return reduction_

    def check_uniqueness(self, timestamps):
        """
        Check the uniqueness of the dataset.

        Parameters
        ----------
        timestamps
            A vector of timestamps. As default, this can be as numpy time array.

        Returns
        ----------
        uniqueness
            Uniqueness of a dataset, as given by the proportion of unique labels to the number of all data points in the set.
        """
        uniqueness = 0
        number_of_instances = timestamps.shape[0]

        unique = np.unique(timestamps)
        number_of_uniques = len(unique)

        if number_of_instances != number_of_uniques:
            uniqueness = (number_of_uniques/number_of_instances)

        return uniqueness

    def check_variance(self, dataset):
        """
        Check variance in time series data for empty columns.

        Parameters
        ----------
        dataset
            A table of data. This should be formatted as times in dimension 0 and features/data as dimension 1.

        Returns
        ----------
        alert_
            Binary alert flag, specifying an anomaly in variance.
        variance_
            The resulting variance of the dataset.
        """
        variance_ = np.zeros(dataset.shape[1])
        alert_ = 0

        for column in range(dataset.shape[1]):
            variance_[column] = np.var(dataset[:, column])
            if variance_[column] == 0:
                print('Feature ', column, ' has 0 variance. Consider dropping it from the set.')
                alert_ = 1
                self.alert_raised_ = 1

        return alert_, variance_

    def check_anomalies(self, dataset):
        """
        Check for outliers in time series data.

        Parameters
        ----------
        dataset
            A table of data. This should be formatted as times in dimension 0 and features/data as dimension 1.

        Returns
        ----------
        outliers_
            Binary variable specifying which data points are likely outliers. This is given as a table NxM, where
            N is the number of data instances and M is the number of features.
        """
        threshold = -1000

        means_ = np.zeros(dataset.shape[1])
        outliers_ = np.zeros((dataset.shape[0], dataset.shape[1]))
        alert_ = 0

        for column in range(dataset.shape[1]):
            means_[column] = np.mean(dataset[:, column])

        covariances_ = np.cov(dataset.T)

        for column in range(dataset.shape[1]):
            for row in range(dataset.shape[0]):
                probs = np.log(self.evaluate_point(dataset[row, column], means_, covariances_))
                if probs < threshold:
                    outliers_[row, column] = 1
                    print('I have found some outlying data on column:', column,'row:', row, 'Consider checking.')
                    alert_ = 1
                    self.alert_raised_ = 1

        return outliers_

    def check_correlations(self, dataset):
        """
        Check the correlations between variables in the dataset.

        Parameters
        ----------
        dataset
            A table of data. This should be formatted as times in dimension 0 and features/data as dimension 1.

        Returns
        ----------
        covariances_
            Covariance of the dataset.
        pearson_
            Pearson coefficient of the feature set, given by MxM, where M is the length of the feature vector.
        spearman_
            Spearman coefficient of the feature set, given by MxM, where M is the length of the feature vector.
        """
        number_of_features = dataset.shape[1]

        covariances_ = np.cov(dataset.T)
        pearson_ = np.zeros((number_of_features, number_of_features))
        spearman_ = np.zeros((number_of_features, number_of_features))

        for outer_ in range(0, number_of_features):
            for inner_ in range(0, number_of_features):

                # check Pearson's
                pearson_[outer_, inner_], _ = stats.pearsonr(dataset[:, outer_], dataset[:, inner_])

                # check Spearman's
                spearman_[outer_, inner_], _ = stats.spearmanr(dataset[:, outer_], dataset[:, inner_])

        return covariances_, pearson_, spearman_
