import numpy
import dateutil
import pandas
from pandas import Timedelta


class RandomTimeSeries(object):
    """

    Parameters
    ----------

    generator_list : (K, ) array_like
        List of K generator functions. Each generator function should return an
        ndarray with shape (N, D) where N is an arbitrary number of samples and
        D is always the same number of feature dimentions.

    labels : (K, ) array_like, optional
        Label corresponding to each of the K generators. The label is not used
        by the object but can be useful as a reference. If None a list of
        integers frm 0 to K is generated.

    priors : (K, ) array_like, optional
        Prior probability for each of the K generators that is used to sample
        from a multinomial distribution.

    samplesize : str
        String of the form 'S', '3S', '5Min', '3H', '1D', '2M', denoting the
        amount of time for each sample.

    Attributes
    ----------

    labels : (K, ) array_like
        Label corresponding to each generator
    """
    def __init__(self, generator_list, labels=None, priors=None,
                 samplesize='1S'):
        self.generator_list = generator_list
        if labels is None:
            labels = numpy.arange(len(generator_list))
        if priors is None:
            priors = numpy.ones(len(generator_list))/len(generator_list)

        assert(len(generator_list) == len(labels))
        assert(len(labels) == len(priors))
        self.labels = labels
        self.priors = numpy.array(priors)/sum(priors)
        self.samplesize = Timedelta(samplesize)

    def generate(self, start_time, end_time):
        """
        Returns a tuple with datetimes, features and labels.

        Parameters
        ----------
        start_time : datetime
            Datetime of the first generated sample

        end_time : datetime
            Datetime of the last generated sample

        Returns
        -------
        ts : (N, ) ndarray of datetime, optional
            One-dimensional array with the datetime of every label.

        X : (N, D) ndarray
            Matrix with N samples and D feature values.

        y_array : (N, ) ndarray of integers
            One-dimensional array with all the labels in numerical discrete format.
        """
        start_time = dateutil.parser.parse(start_time)
        end_time = dateutil.parser.parse(end_time)
        total_samples = int((end_time - start_time)/self.samplesize)
        current_samples = 0
        X = []
        y = []
        while current_samples < total_samples:
            current_class = numpy.random.multinomial(1, self.priors).argmax()
            sample = self.generator_list[current_class]()
            y.append(numpy.ones(len(sample)) * current_class)
            X.append(sample)
            current_samples += len(sample)

        ts = pandas.date_range(start_time, periods=total_samples,
                               freq=self.samplesize)

        return (ts, numpy.concatenate(X)[:total_samples],
                numpy.concatenate(y)[:total_samples])


if __name__ == '__main__':
    # First example
    generator_list = [lambda: numpy.random.randn(numpy.random.randint(1, 100), 3)/2,
                      lambda: numpy.random.randn(numpy.random.randint(15, 300), 3),
                      lambda: numpy.random.randn(numpy.random.randint(30, 90), 3)*2,
                      lambda: numpy.random.randn(numpy.random.randint(400, 600), 3)/16]

    labels = ['stand', 'walk', 'run', 'sleep']
    rts = RandomTimeSeries(generator_list, labels=labels,
                           priors=[3/6, 1/6, 1/6, 1/6], samplesize='1Min')

    ts, X, y = rts.generate('11/06/2019', '11/07/2019')
    print(len(ts), 'samples have been generated')
    print(ts)
    print(y)

    # Second example
    def sin_gaussian_rand(n_samples, size):
        sin_signal = numpy.sin(numpy.repeat(numpy.arange(n_samples),
                                            size).reshape(n_samples, size))
        return sin_signal + numpy.random.randn(n_samples, size)

    generator_list = [lambda: numpy.random.randn( numpy.random.randint(30, 90), 4) + [3, 1, 0, 0],
                      lambda: sin_gaussian_rand(numpy.random.randint(15, 60), 4) + [1, 3, 1, 0],
                      lambda: numpy.random.randn(numpy.random.randint(100, 400), 4) + [0, 0, 4, 2]]

    labels = ['kitchen', 'livingroom', 'bedroom']
    rts = RandomTimeSeries(generator_list, labels=labels,
                                                  priors=[2/5, 2/5, 1/5],
                           samplesize='1Min')

    ts, X, y = rts.generate('07:00 11-06-2019', '23:30 11-06-2019')
    print(len(ts), 'samples have been generated')
    print(ts)
    print(y)

    # Third example
    generator_list = [lambda: numpy.random.randn(numpy.random.randint(1, 3), 3)/2,
                      lambda: sin_gaussian_rand(numpy.random.randint(1, 3), 3),
                      lambda: numpy.random.dirichlet([0.2, 0.5, 0.2], size=numpy.random.randint(1, 2)),
                      lambda: numpy.random.rand(numpy.random.randint(5, 12), 3)/16]

    labels =['stand', 'walk', 'run', 'sleep']
    rts = RandomTimeSeries(generator_list, labels=labels,
                           priors=[3/6, 1/6, 1/6, 1/6], samplesize='10Min')

    ts, X, y = rts.generate('11/06/2019', '11/11/2019')
    print(len(ts), 'samples have been generated')
    print(ts)
    print(y)
