#!/usr/bin/env python

"""expstoreprint.py: Exporting of experience store contents."""

import numpy as np
import matplotlib.pyplot as plt


def export_exp_store(exp_store_percepts,
                     exp_store_actions,
                     exp_store_rewards,
                     filepath,
                     state_stm=1,
                     lower=0,
                     upper=-1,
                     percept_width=64,
                     percept_height=64):
    if upper == -1: upper = len(exp_store_rewards) - 1
    for i in xrange(lower + state_stm - 1, upper):
        fig = plt.figure(figsize=(3, state_stm))
        axs = []
        for j in xrange(0, state_stm):
            percept_before = 1 - exp_store_percepts[i - j]
            percept_after = 1 - exp_store_percepts[i + 1 - j]

            # print percept_before[]

            axs.append(fig.add_subplot(state_stm, 3, 3 * j + 1, frameon=False))
            axs[-1].imshow(percept_before.reshape(percept_height,
                                                  percept_width,
                                                  -1),

                           interpolation='nearest')

            axs.append(fig.add_subplot(state_stm, 3, 3 * j + 2, frameon=False))
            axs[-1].imshow(percept_after.reshape(percept_height,
                                                 percept_width,
                                                 -1),
                           interpolation='nearest')

        axs.append(fig.add_subplot(state_stm, 3, 3 * state_stm, frameon=False))
        axs[-1].text(0, 0, 'action: ' + str(exp_store_actions[i]))
        axs[-1].text(0, 0.2, 'reward: ' + str(exp_store_rewards[i]))

        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        fig.savefig(filepath + str(i) + '.png', dpi=300, bbox_inches='tight')
