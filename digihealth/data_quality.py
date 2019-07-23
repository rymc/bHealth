import numpy as np
import pandas as pd
from scipy import stats

class DataQuality:

    def __init__(self, time_cols, label_cols, feature_cols, sample_rate):

        self.timestamp_columns = time_cols
        self.label_columns = label_cols
        self.feature_columns = feature_cols
        self.sample_rate = sample_rate

        self.alert_raised_ = 0


    def evaluate_point(self, point, means, covariances):
        m_dist_x = np.dot((point - means).transpose(), np.linalg.inv(covariances))
        m_dist_x = np.dot(m_dist_x, (point - means))
        prob = 1 - stats.chi2.cdf(m_dist_x, 3)
        return prob

    def check_continuity(self, timestamps):
        """Check for continuity in time series data."""
        df_time = timestamps.values.astype('datetime64')
        number_of_instances_prior = len(df_time)
        x = np.ones(number_of_instances_prior)
        df_time = pd.DataFrame({'Time': df_time})

        possible_times = pd.date_range(start=pd.to_datetime(df_time['Time'][0]), end=pd.to_datetime(df_time['Time'][number_of_instances_prior-1]), freq=self.sample_rate).time
        df_time = df_time.set_index('Time').reindex(possible_times).reset_index().reindex(columns=df_time.columns)
        df_time = df_time.ffill()

        number_of_instances_post = len(df_time)
        reduction = (number_of_instances_post - number_of_instances_prior)/number_of_instances_post

        if reduction == 0:
            alert_ = 0
        else:
            alert_ = 1
            self.alert_raised_ = 1
            red = reduction * 100
            print('The set appears discontinuous. From what I can see, there is almost', red, 'percent missing. Check for completeness of the data.')

        return alert_, reduction

    def check_variance(self, dataset):

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

        return alert_, outliers_
