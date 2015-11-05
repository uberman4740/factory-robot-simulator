import numpy as np
import matplotlib.pyplot as plt


class World(object):
    def __init__(self, size, n_boxes, box_size, box_speedlimits, rng, dtype='float32'):
        self.size = size
        self.n_boxes = n_boxes
        self.rng = rng
        self.box_speedlimits = box_speedlimits[0], box_speedlimits[1] + 1
        self.box_size = box_size
        # self.box_positions = rng.randint(0, size, size=n_boxes)
        # self.box_velocities = rng.randint(*box_speedlimits, size=n_boxes)
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.box_positions = self.rng.randint(0, self.size, size=self.n_boxes)
        self.box_velocities = self.rng.randint(*self.box_speedlimits, size=self.n_boxes)

    def step(self):
        self.box_positions = (self.box_positions + self.box_velocities) % self.size

    def get_window(self, position, window_size):
        world_state = np.zeros(self.size)
        for b in self.box_positions:
            world_state[b: b + self.box_size] = 1
            if self.size - b < self.box_size:
                world_state[: b + self.box_size - self.size] = 1

        if position + window_size < self.size:
            window = world_state[position: position + window_size]
        else:
            window = np.append(world_state[position:], world_state[:window_size - (self.size - position)])
        return np.asarray(window, dtype=self.dtype)


def create_seq_data(world_generator, window_size, sequence_length, n_sequences, rng, dtype='float32'):
    xs = []
    ys = []
    for i in xrange(n_sequences):
        world = world_generator()
        percepts, actions = create_sequence(world, window_size, sequence_length + 1, rng)
        x = np.append(percepts, actions.reshape(-1, 1), axis=1)
        xs.append(x[:-1])
        ys.append(percepts[1:])
    return np.asarray(xs, dtype=dtype), np.asarray(ys, dtype=dtype)


def create_seq_data_complex(world, window_size, sequence_length, n_sequences, rng, action_duration,
                            dtype='float32', return_world=False):
    xs = []
    ys = []
    ws = []
    ps = []
    for i in xrange(n_sequences):
        world.reset()
        percepts, actions, w, p = create_sequence_complex(world, window_size, sequence_length + 1, action_duration, rng,
                                                          return_world)
        x = np.append(percepts, actions.reshape(-1, 1), axis=1)
        xs.append(x[:-1])
        ys.append(percepts[1:])
        ws.append(w)
        ps.append(p)
    if return_world:
        return (np.asarray(xs, dtype=dtype),
                np.asarray(ys, dtype=dtype),
                np.asarray(ws, dtype=dtype),
                np.asarray(ps, dtype='int16'))
    return np.asarray(xs, dtype=dtype), np.asarray(ys, dtype=dtype)


def iid_world_generator(world_size, p, rng):
    def generator():
        return rng.choice([0, 1], p=[1 - p, p], size=world_size)

    return generator


def bigbox_world_generator(world_size, box_size, box_min_distance, mean_box_number, rng):
    slot_size = box_size + box_min_distance
    n_slots = world_size / slot_size
    assert mean_box_number <= n_slots

    def generator():
        p = 1. * mean_box_number / n_slots
        boxes = rng.choice([0, 1], p=[1 - p, p], size=n_slots)
        world = np.zeros(world_size)
        for i in xrange(box_size):
            world[i::(box_size + box_min_distance)][:n_slots] = boxes
        return np.append(world, np.zeros(world_size % slot_size))

    return generator


def create_sequence(world, window_size, sequence_length, rng):
    current_pos = rng.randint(0, len(world))
    sequence = []
    actions = []
    for i in xrange(sequence_length):
        if current_pos + window_size < len(world):
            current_window = world[current_pos: current_pos + window_size]
        else:
            current_window = np.append(world[current_pos:], world[:window_size - (len(world) - current_pos)])

        sequence.append(current_window)
        action = 2 * rng.randint(0, 2) - 1
        current_pos = (current_pos + action) % len(world)
        actions.append(action)
    return np.asarray(sequence), np.asarray(actions)


def create_sequence_complex(world, window_size, sequence_length, action_duration, rng, return_world):
    """ Sequences with three actions. """
    current_pos = rng.randint(0, world.size)
    window_seq = []
    actions = []
    world_seq = []
    position_seq = []

    remaining_action_length = 0
    for i in xrange(sequence_length):
        if remaining_action_length == 0:
            current_action = rng.randint(0, 3) - 1
            remaining_action_length = rng.randint(*action_duration)

        world.step()
        window_seq.append(world.get_window(current_pos, window_size))
        if return_world:
            world_seq.append(world.get_window(0, world.size))
            position_seq.append(current_pos)
        current_pos = (current_pos + current_action) % world.size
        actions.append(current_action)
        remaining_action_length -= 1
    return np.asarray(window_seq), np.asarray(actions), np.asarray(world_seq), np.asarray(position_seq)


def bernoulli_array(ps, rng):
    return np.asarray([rng.choice([0, 1], p=[1 - p, p]) for p in ps])


def recursive_prediction(model, past_sequence, subsequent_actions, pred_mode, rng=None):
    assert rng is not None or pred_mode != 'sample'

    prediction = []
    for i in xrange(len(subsequent_actions)):
        current_prediction = model.predict(past_sequence[np.newaxis, :, :])[0, -1]
        if pred_mode == 'round':
            current_prediction = np.round(current_prediction)
        elif pred_mode == 'sample':
            current_prediction = bernoulli_array(current_prediction, rng)
        elif pred_mode == 'none':
            pass
        else:
            raise Exception('Illegal pred mode.')

        # print past_sequence.shape
        # print current_prediction.shape
        # print subsequent_actions[i].shape
        # print np.append(current_prediction, subsequent_actions[i]).shape
        past_sequence = np.append(past_sequence,
                                  np.append(current_prediction, subsequent_actions[i])[np.newaxis, :],
                                  axis=0)
        prediction.append(current_prediction)
    return np.asarray(prediction)


# --- Plotting ---
def plot_percept(ax, percept, y_offset=0, color='#003377', height_scale=1.0):
    rects = ax.bar(np.arange(len(percept)),
                   height=height_scale*np.ones_like(percept),
                   bottom=y_offset,
                   lw=0,
                   color=color,
                   fill=True,
                   width=1.05,
                   alpha=1.0)
    for i, r in enumerate(rects):
        r.set_alpha(percept[i])
    ax.set_ylim(0, 1 + y_offset)
    ax.set_xlim(0, len(percept))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


def draw_arrow(ax, direction):
    hide_axes(ax)
    ax.set_frame_on(False)
    ax.arrow(0.5, 0.5, 0.2 * direction, 0.1 * (1 - np.abs(direction)),
             head_width=0.1, head_length=0.05, fc='k', ec='k')


def hide_axes(ax):
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


def plot_percepts(percepts, actions=None, predictions=None, box_width=1):
    assert predictions is None or percepts.shape == predictions.shape
    assert actions is None or percepts.shape[0] == actions.shape[0]

    fig_width = 0.5 * len(percepts[0]) / box_width
    fig_height = 0.5 * len(percepts) if predictions is None else len(percepts)
    if actions is not None:
        fig_height *= 2

    fig = plt.figure(figsize=(fig_width,
                              fig_height))
    for i, percept in enumerate(percepts):
        if actions is not None:
            ax = fig.add_subplot(2 * len(percepts), 1, 2 * i + 1)

            # Arrows to indicate action
            ax_action = fig.add_subplot(2 * len(percepts), 1, 2 * i + 2)
            # ax_action.arrow(0.5, 0.5, 0.2*actions[i], 0.1*(1 - np.abs(actions[i])),
            #                 head_width=0.1, head_length=0.05, fc='k', ec='k')
            draw_arrow(ax_action, actions[i])

        else:
            ax = fig.add_subplot(len(percepts), 1, i + 1)
        plot_percept(ax, percept)
        if predictions is not None:
            plot_percept(ax, predictions[i], 1, color='#007733')
    return fig
