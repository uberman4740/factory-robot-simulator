import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import numpy as np

import threading
import sys

COLOR_TABLE = [(200, 50, 120),
               (10, 80, 230),
               (30, 180, 130),
               (150, 150, 150)]


class LiveCharting(object):
    def __init__(self,
                 n_curves,
                 y_min,
                 y_max,
                 curve_width=3,
                 steps=400,
                 title=None,
                 ylabel=None):
        self.steps = steps
        self.p = pg.plot()
        if title is not None:
            self.p.setWindowTitle(title)
        self.p.setRange(QtCore.QRectF(0, y_min, self.steps, y_max-y_min))
        if ylabel is not None:
            self.p.setLabel('left', ylabel)
        self.curves = [self.p.plot(pen=pg.mkPen(
            color=COLOR_TABLE[i % len(COLOR_TABLE)],
            width=curve_width))
                       for i in xrange(n_curves)]

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)
        self.X = np.linspace(0, 8, self.steps)
        # self.data = np.random.normal(size=5000)
        self.data = [np.zeros_like(self.X) for i in xrange(n_curves)]

    def set_current_value(self, index, value):
        self.data[index][:-1] = self.data[index][1:]
        self.data[index][-1] = value

    def update(self):
        for curve, d in zip(self.curves, self.data):
            curve.setData(d)

    @staticmethod
    def run():
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
