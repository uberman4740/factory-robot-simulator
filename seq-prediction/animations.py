import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from reduceddynamics import plot_percept, draw_arrow

import numpy as np


def make_anim(filename, pre_percepts, post_percepts, predictions, actions, world_states=None, positions=None,
              box_width=1):
    assert ((world_states is None and positions is None) or
            (world_states is not None and positions is not None))

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='title', artist='artist',
                    comment='comment')
    writer = FFMpegWriter(fps=3, metadata=metadata)

    if world_states is not None:
        assert world_states.shape[0] == actions.shape[0]
        assert world_states.shape[0] == positions.shape[0]
        window_size = pre_percepts.shape[1]
        world_height_scale = 1.* window_size / world_states.shape[1]
        world_offset = 1
    else:
        world_offset = 0

    if len(predictions.shape) == 2:
        assert predictions.shape == post_percepts.shape
        n_prediction_plots = 1
    elif len(predictions.shape) == 3:
        print predictions.shape, post_percepts.shape
        assert predictions.shape[1:] == post_percepts.shape
        n_prediction_plots = predictions.shape[0]
    else:
        raise Exception('Wrong format for predictions.')

    fig_width = 2 * 0.5 * len(pre_percepts[0]) / box_width
    fig_height = 2 * (n_prediction_plots + world_offset)
    fig = plt.figure(figsize=(fig_width, fig_height))

    def _plot_percept(percept, pos, height_scale=1.0, window=None, color='#003377'):
        ax = fig.add_subplot(n_prediction_plots + 2 + world_offset,
                             1,
                             pos)
        plot_percept(ax, percept, height_scale=height_scale, color=color)
        if window is not None:
            # ax.bar(window[])
            bar_kwargs = {'alpha': 0.7, 'color': '#0050a0', 'width': 1.0, 'lw': 0}
            position = window[0]
            w_size = window[1]

            if position + w_size <= percept.shape[0]:
                ax.bar(np.arange(position),
                       np.ones(position),
                       **bar_kwargs)
                ax.bar(np.arange(position+w_size, percept.shape[0]),
                       np.ones(percept.shape[0] - (position+w_size)),
                       **bar_kwargs)
            else:
                ax.bar(np.arange(position+w_size - percept.shape[0], position),
                       np.ones(percept.shape[0] - w_size),
                       **bar_kwargs)

    def _draw_arrow(action, pos):
        ax_arrow = fig.add_subplot(n_prediction_plots + 2 + world_offset,
                                   1,
                                   pos)
        draw_arrow(ax_arrow, action)

    with writer.saving(fig, filename, 100):
        for i in range(len(pre_percepts)):
            fig.clear()
            if world_states is not None:
                _plot_percept(world_states[i], pos=1,
                              height_scale=world_height_scale,
                              window=(positions[i], window_size))
            _plot_percept(pre_percepts[i], pos=2 + world_offset)
            _draw_arrow(actions[i], pos=1 + world_offset)
            writer.grab_frame()

        for i in range(len(post_percepts)):
            fig.clear()
            if world_states is not None:
                _plot_percept(world_states[i + len(pre_percepts)], pos=1,
                              height_scale=world_height_scale,
                              window=(positions[i + len(pre_percepts)], window_size))
            _plot_percept(post_percepts[i], pos=2 + world_offset)
            _draw_arrow(actions[i + len(pre_percepts)], pos=1 + world_offset)

            if len(predictions.shape) == 3:
                for j in xrange(predictions.shape[0]):
                    _plot_percept(predictions[j, i], pos=3 + j + world_offset, color='#007733')
            elif len(predictions.shape) == 2:
                _plot_percept(predictions[i], pos=3 + world_offset, color='#007733')

            writer.grab_frame()

