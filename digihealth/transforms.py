import numpy as np
import scipy.stats
from scipy.signal import butter, filtfilt

class Transforms:


    def __init__(self, window_length, window_overlap):
        self.window_length = window_length
        self.current_position = 0
        self.window_overlap = window_overlap

    @staticmethod
    def zero_crossings(x):
        """Return the number of zero crossings."""
        sign = [1,0]
        direction = 0
        count_zc = 0
        if x[0] >= 0:
            direction = 1

        for i in range(len(x)):
            if (x[i] >= 0 and direction == 0) or (x[i] < 0 and direction == 1):
                direction = sign[direction]
                count_zc += 1
        return count_zc

    @staticmethod
    def mean_crossings(x):
        """Return the number of mean crossings."""
        x = x - np.mean(x)
        return Transforms.zero_crossings(x)

    @staticmethod
    def interq(x):
        """Return the interquartile range."""
        interquartile = scipy.stats.iqr(x)
        return interquartile

    @staticmethod
    def skewn(x):
        """Return the skewness."""
        skewness = scipy.stats.mstats.skew(x)
        skewness = skewness.data.flatten()[0]
        return skewness

    @staticmethod
    def spec_energy(x):
        """Return the spectral energy."""
        f = np.fft.fft(x)
        F = abs(f)
        return sum(np.square(F))

    @staticmethod
    def spec_entropy(x):
        """Return the spectral entropy."""
        f = np.fft.fft(x)
        F = abs(f)
        sumf = sum(F)
        if sumf == 0:
            sumf = 1
        nf = F/sumf
        min_nf = 1
        if (min(nf) != max(nf)) and (min(nf) != 0):
            min_nf = min(m for m in nf if m > 0)

        logf = np.log((nf+min_nf))
        spectral_entropy = -1*sum(nf*logf)
        return spectral_entropy

    @staticmethod
    def p25(x):
        """Return the 25th-percentile."""
        return np.percentile(x, 25)

    @staticmethod
    def p75(x):
        """Return the 75th-percentile."""
        return np.percentile(x, 75)

    @staticmethod
    def kurtosis(x):
        """Return the kurtosis."""
        return scipy.stats.kurtosis(x, fisher=False, bias=True)
    
    def _butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    @staticmethod
    def butter_lowpass_filter(x, cutoff, fs, order=5):
        b, a = _butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, x)
        return y

    def slide(self, x, update=True):
        """Slide and return the window of data."""
        window = x[self.current_position-self.window_length:self.current_position]
        if len(window) > 0:
            if len(window.shape) > 1:
                window = window[~np.isnan(window).any(axis=1)]
            else:
                window = window[~np.isnan(window)]
        if update:
            self.current_position += self.window_overlap
        return window
