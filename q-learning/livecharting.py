#!/usr/bin/env python

"""livecharting.py: provides interface to display charts in real-time."""

import numpy as np
import time
import matplotlib
import threading

# Use backend that works with Mac OS
# matplotlib.use('TkAgg')
matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

plt.style.use(['dark_background'])


# Animation parameters
min_speed = 0.1
max_speed = 1.0


class LiveCharting(object):
    def __init__(self,
                 n_curves,
                 ymin,
                 ymax,
                 x_resolution=100,
                 ylabel=None,
                 data_labels=None,
                 xlabel='time'):
        # Setting up figure
        self.fig = plt.figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylim(ymin, ymax)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)

        self.ax.set_xlabel(xlabel)

        self.X = np.linspace(-6, 0, x_resolution)
        self.YS = [np.zeros_like(self.X) for i in xrange(n_curves)]

        # Plot curves
        if data_labels is not None:
            self.plots = [self.ax.plot(self.X, Y, linewidth=2, label=lbl)
                          for lbl, Y in zip(data_labels, self.YS)]
            self.ax.legend(loc=2)
        else:
            self.plots = [self.ax.plot(self.X, Y, linewidth=2) for Y in self.YS]

        self.animation = FuncAnimation(self.fig,
                                       self.update,
                                       interval=10)

    def set_current_value(self, index, value):
        self.YS[index][:-1] = self.YS[index][1:]
        self.YS[index][-1] = value

    def update(self, num):
        for ((p,), Y) in zip(self.plots, self.YS):
            p.set_data(self.X, Y)
