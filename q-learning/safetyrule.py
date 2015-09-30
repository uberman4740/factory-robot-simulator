
import numpy as np

class SafetyRule(object):

    def __init__(self, vector_start, vector_stop, threshold, safety_action):
        self.vector_start = vector_start
        self.vector_stop = vector_stop
        self.threshold = threshold
        self.safety_action =  safety_action


    def check_percept(self, percept):
        if np.mean(percept[self.vector_start: self.vector_stop]) > self.threshold:
            return self.safety_action
        else:
            return None