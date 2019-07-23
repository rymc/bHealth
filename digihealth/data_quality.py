import numpy as np
import pandas as pd
import scipy.stats

class DataQuality:

    def __init__(self, time_cols, label_cols, feature_cols, sample_rate):

        self.timestamp_columns = time_cols
        self.label_columns = label_cols
        self.feature_columns = feature_cols
        self.sample_rate = sample_rate

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

        if reduction != 0:
            continuity = 0
        else:
            continuity = 1

        return continuity, reduction

    #
    # def check_realness(self, dataset):
    #
    # def check_variance(self, dataset):
    #
    # def check_noise(self, dataset):
    #
    # def check_label_space(self, dataset):
    #
    # def check_imputation_potential(self, dataset):