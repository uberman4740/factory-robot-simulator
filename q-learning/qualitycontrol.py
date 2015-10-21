import cPickle
import numpy as np


class QualityLogger(object):
    def __init__(self, filename, quality_logger=None, log_positive_diffs=True):
        self.filename = filename
        self.log_positive_diffs=log_positive_diffs
        self.last_value = 0
        self.total = 0
        self.time = 0
        self.quality_logger = quality_logger

    def set_value(self, value, time):
        diff = value - self.last_value
        if self.log_positive_diffs and diff > 0:
            self.total += diff
        self.time = time
        self.last_value = value
        self.save(filename=self.filename)

    @classmethod
    def load(cls, filename):
        f = open(filename)
        return cPickle.load(f)

    def save(self, filename):
        f = open(filename, 'w')
        cPickle.dump(self, f)
        f.close()

    def get_sigma(self):
        if self.quality_logger is not None:
            rate = 1.0 * self.quality_logger.total / self.quality_logger.time
            expected = rate * self.time
            diff = self.value - expected
            return diff / 3. * np.sqrt(expected)
        else:
            return None


