
import numpy as np
import matplotlib.pyplot as plt


class LiveBarPlotter(object):

    def __init__(self, n_categories, n_bars_per_category):
        self.n_categories = n_categories
        self.n_bars_per_category = n_bars_per_category
        self.chart_fig = plt.figure(figsize=(5.0, 3.0))
        self.chart_ax = self.chart_fig.add_subplot(111)
        self.chart_ax.set_xlim(-n_categories/2.0, n_categories/2.0)
        self.chart_ax.set_ylim(0.0, 1.0)
        bar_width = 0.1

        chart_colors = ['#aa2222', '#33eecc', '#55aa22', '#ddaa33', '#999999']
        self.charts = [self.chart_ax.bar(np.arange(n_categories)-(n_categories/2) + \
                                    (i - n_bars_per_category/2.0 + 0.5)*bar_width,
                                    np.ones(n_categories),
                                    bar_width,
                                    color=chart_colors[i],
                                    align='center')
                       for i in xrange(n_bars_per_category)]
        self.chart_fig.canvas.draw()
        plt.show(block=False)


    def update(self, data):
        for chart, ds in zip(self.charts, data.reshape(self.n_bars_per_category,
                                                     self.n_categories)):
            for rect, d in zip(chart, ds):
                rect.set_height(d)
        self.chart_fig.canvas.draw()
