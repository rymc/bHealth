import pandas
import re
import os
import csv
import numpy

from scipy.stats import friedmanchisquare

from datawrapper.house import HouseVisit

def df_start_end_to_time(time, df, remove_se=True):
    """ Transforms a DataFrame with start and end to a time serie
    Parameters:
    -----------
        time: pandas.tseries.tdi.TimedeltaIndex
            This is the time instances that we are interested on sampling
        df: pandas.core.frame.DataFrame
            This dataframe contains start and end columns and optionally
            additional columns.
        remove_se: bool
            If true, removes the columns start and end
    Returns:
    --------
        df: pandas.core.frame.DataFrame
            This dataframe has as a index the original time steps and all the
            columns of the initial DataFrame where in each row the index is
            contained between start and end.
    """
    # Create a new pandas DataFrame to join with
    # Create a dummy column to merge
    df_time = pandas.DataFrame(data={'time': time, 'dummy': 1})
    df['dummy'] = 1

    # FIXME this is really memory expensive (look for alternative)
    # - cons: always big, independent of the resampling
    # This operation removes all the time rows that are not between start and
    # end
    cross_join = df_time.merge(df, how='left', on='dummy')
    del df['dummy']

    # This operation removes all the time rows that are not between start and
    # end
    cond_join = cross_join[(cross_join['start'] <= cross_join['time']) &
                           (cross_join['end'] >= cross_join['time'])]

    # We don't need the cross join anymore
    del cross_join
    # We don't need dummy anymore
    del cond_join['dummy']
    del df_time['dummy']

    # Get the lost rows back
    cond_join = df_time.merge(cond_join, how='left', on='time')

    # Use the time as index
    cond_join.index = cond_join['time']
    del cond_join['time']

    if remove_se:
        del cond_join['start']
        del cond_join['end']

    return cond_join


def get_all_paths(path, path_expression=""):
    '''Returns a list of tuples (root, subdirs, files) where every file is
    filtered by the given expression
    '''
    regexp = re.compile(path_expression)
    file_paths = []
    for root, subdirs, files in sorted(os.walk(path, followlinks=True)):
        #from IPython import embed; embed()
        #file_list = list(filter(regexp.match, files))
        if regexp.match(root):
            # TODO add the matching patters from the regexp (eg. year, month, day)
            file_paths.append(dict(root = root, subdirs = subdirs,
                                  file_list = files))

    return file_paths
