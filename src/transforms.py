import numpy as np

class Transforms:


    def __init__(self, window_length, window_overlap):
        self.window_length = window_length
        self.current_position = 0
        self.window_overlap = window_overlap

    @staticmethod
    def zero_crossings(x):
        sign = [1,0]
        direction = 0
        count_zc = 0;
        if x[0] >= 0:
            direction = 1

        for i in range(len(x)):
            if (x[i] >= 0 and direction == 0) or (x[i] < 0 and direction == 1):
                direction = sign[direction]
                count_zc += 1
        return count_zc

    @staticmethod
    def mean_crossings(x):
        x = x - np.mean(x)
        return Transforms.zero_crossings(x)

    @staticmethod
    def interq(x):
        p25 = np.percentile(x, 25)
        p75  = np.percentile(x, 75)
        interquartile = p25 - p75
        return interquartile

    @staticmethod
    def skewn(x):
        skewness = scipy.stats.mstats.skew(x)              
        skewness = skewness.data.flatten()[0]
        return skewness

    @staticmethod
    def spec_energy(x):
        f = np.fft.fft(x)  
        F = abs(f)
        return sum(np.square(F))  

    @staticmethod
    def spec_entropy(x):
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
        return np.percentile(x, 25)

    @staticmethod
    def p75(x):
        return np.percentile(x, 75)

    @staticmethod
    def kurtosis(x):
        return scipy.stats.mstats.kurtosis(x)

    def slide(self, x, update=True):
        window = x[self.current_position-self.window_length:self.current_position]
        if len(window) > 0:
            if len(window.shape) > 1:
                window = window[~np.isnan(window).any(axis=1)]
            else:
                window = window[~np.isnan(window)]
        if update:
            self.current_position += self.window_overlap
        return window
