#!/usr/bin/env python

"""maincontroller.py: Central management of Q-Learning data flow and communication."""

import socket
import time

import numpy as np

from q_learner import QLearner, QNetwork
import inputlistener
from sensordecoder import SensorDecoder
import params as ps
from labeling_network import FullyConnectedLayer, linear, Network
from log import log


class MainController(object):
    RECEIVE, SEND, LEARN = range(3)

    def __init__(self,
                 q_learner,
                 sensor_decoder,
                 state_encoder_fn,
                 timeout_period,
                 remote_host,
                 remote_port,
                 learning_rate,
                 learning_iterations_per_step,
                 random_action_duration,  # in number of frames.
                 epsilon_decrease_duration,  # in number of frames.
                 epsilon_start=1.0,
                 epsilon_end=0.0,
                 burn_in=100,
                 extensions=[],  # plotting, etc
                 frame_counter_increment=1,
                 prng=None):
        self.frame_counter = 0
        self.total_steps = 0
        self.frame_counter_increment = frame_counter_increment
        self.timeout_period = timeout_period
        self.last_send_realtime = 0.0
        self.q_learner = q_learner
        self.sensor_decoder = sensor_decoder
        self.state_encoder_fn = state_encoder_fn
        self.current_state = MainController.RECEIVE
        self.current_decision = None
        self.sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.learning_rate = learning_rate
        self.learning_iterations_per_step = learning_iterations_per_step
        if prng is not None:
            self.prng = prng
        else:
            self.prng = np.random.RandomState()
        self.randaction_duration = random_action_duration
        self.current_randaction = None
        self.current_randaction_termination = -1
        self.epsilon_decrease_duration = epsilon_decrease_duration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.burn_in = burn_in
        self.current_total_reward = 0

    def do(self):
        """Check for data receipt and send decisions to remote host. """
        if self.current_state == MainController.RECEIVE:
            if self.timeout():
                log('Timeout.')
                self.current_state = MainController.SEND
            else:
                data = inputlistener.collect_current_data()
                if len(data) > 0:
                    log('Received data ({0}):'.format(len(data)))
                    for packet in data:
                        self.sensor_decoder.add_data(packet, self.frame_counter)
                    if self.sensor_decoder.state_info_complete():
                        percept, prev_action, total_reward = self.sensor_decoder. \
                            get_current_data()
                        self.remember_percept(percept,
                                              prev_action,
                                              total_reward - self.current_total_reward)
                        self.current_total_reward = total_reward
                        self.current_decision = self.get_decision()
                        self.advance_frame()
                        self.current_state = MainController.SEND

        if self.current_state == MainController.SEND:
            log('Sending... (total_steps={0}, current_decision={1})'.format(self.total_steps,
                                                                            self.current_decision))
            self.send_decision(self.current_decision)
            self.current_state = MainController.LEARN

        if self.current_state == MainController.LEARN:
            log('Training...')
            if self.total_steps > self.burn_in:
                for i in xrange(self.learning_iterations_per_step):
                    self.q_learner.train_q_function(self.learning_rate)
                log('qs: {0}'.format(self.q_learner.get_current_qs()), 1)

            log('Training steps completed.')
            self.current_state = MainController.RECEIVE

    def remember_percept(self, percept, last_action, previous_reward):
        """Encode raw percept and pass encoding, last action and reward to Q-Learner."""
        encoding = self.state_encoder_fn(percept)
        self.q_learner.add_observation(encoding.astype(np.float32),
                                       last_action,
                                       previous_reward)

    def get_decision(self):
        """Use Q-Function to decide for current action in epsilon-greedy way."""
        epsilon = max(self.epsilon_end,
                      self.epsilon_start -
                      (self.epsilon_start - self.epsilon_end) *
                      1. * self.total_steps / self.epsilon_decrease_duration)
        log('epsilon: {0:.4}'.format(epsilon), 1)
        if self.current_randaction_termination > self.total_steps:
            return self.frame_counter, self.current_randaction
        if self.prng.uniform(0, 1) >= epsilon \
                and self.q_learner.exp_store_current_size > self.burn_in:
            action = self.q_learner.get_current_best_action()
            self.current_randaction = None
        else:
            self.current_randaction = self.prng.randint(self.q_learner.
                                                        q_function.n_out)
            action = self.current_randaction
            self.current_randaction_termination = self.total_steps + self.randaction_duration
        return self.frame_counter, action

    def send_decision(self, decision):
        """Send (frame_counter, action)-tuple to remote host."""
        self.last_send_realtime = time.time()
        if decision is None:
            log('send_decision: decision is None')
            return
        prev_frame_counter, action = decision
        data = [prev_frame_counter, action]
        self.send_data(data)

    def send_data(self, l, send_checksum=False):
        """Send list of numbers to the stored remote host."""
        if send_checksum:
            checksum = np.mod(np.sum(l), 256)
            l.append(checksum)
        self.sender_socket.sendto(''.join([chr(int(x) % 256) for x in l]),
                                  (self.remote_host, self.remote_port))

    def timeout(self):
        """Determines whether a timeout has taken place."""
        return time.time() - self.last_send_realtime > self.timeout_period

    def advance_frame(self):
        """Updates the step counter and frame counter byte."""
        self.frame_counter = (self.frame_counter + self.frame_counter_increment) % 256
        self.total_steps += 1


def load_labeling_function(filename, mb_size):
    return Network.load_from_file(filename, mb_size)


def load_q_network(filename, state_stm, percept_length,
                   q_hidden_neurons, n_actions, mb_size):
    if filename is not None:
        return QNetwork.load_from_file(filename, mb_size)
    else:
        hidden_layer = FullyConnectedLayer(state_stm * percept_length,
                                           q_hidden_neurons)
        output_layer = FullyConnectedLayer(q_hidden_neurons,
                                           n_actions,
                                           activation_fn=linear)
        return QNetwork([hidden_layer, output_layer], minibatch_size=mb_size)


def main():
    # gc.disable()
    prng = np.random.RandomState(ps.PRNG_SEED)
    sensor_decoder = SensorDecoder(n_fragments=ps.N_FRAGMENTS,
                                   n_checksum_bytes=ps.N_CHECKSUM_BYTES,
                                   frame_counter_position=ps.FRAME_COUNTER_POS,
                                   fragment_id_position=ps.FRAGMENT_ID_POS,
                                   img_data_position=ps.IMG_DATA_POS,
                                   img_fragment_length=ps.IMG_FRAGMENT_LENGTH,
                                   action_position=ps.ACTION_POS,
                                   reward_position=ps.REWARD_POS,
                                   n_reward_bytes=ps.N_REWARD_BYTES)

    labeling_net = load_labeling_function(ps.LABELING_NETWORK_FILE_NAME,
                                          ps.MB_SIZE)
    state_encoder_fn = labeling_net.get_single_output

    q_function = load_q_network(ps.Q_NETWORK_LOAD_FILENAME,
                                ps.STATE_STM,
                                ps.PERCEPT_LENGTH,
                                ps.Q_HIDDEN_NEURONS,
                                ps.N_ACTIONS,
                                ps.MB_SIZE)
    q_learner = QLearner(q_function,
                         exp_store_size=ps.EXP_STORE_SIZE,
                         percept_length=ps.PERCEPT_LENGTH,
                         n_actions=ps.N_ACTIONS,
                         state_stm=ps.STATE_STM,
                         gamma=ps.GAMMA,
                         minibatch_size=ps.MB_SIZE,
                         prng=prng)

    main_controller = MainController(q_learner,
                                     sensor_decoder=sensor_decoder,
                                     state_encoder_fn=state_encoder_fn,
                                     timeout_period=ps.TIMEOUT_PERIOD,
                                     remote_host=ps.REMOTE_HOST,
                                     remote_port=ps.REMOTE_PORT,
                                     learning_rate=ps.LEARNING_RATE,
                                     learning_iterations_per_step=ps.LEARNING_ITERATIONS_PER_STEP,
                                     random_action_duration=ps.RANDOM_ACTION_DURATION,
                                     epsilon_decrease_duration=ps.EPSILON_DECREASE_DURATION,
                                     epsilon_start=ps.EPSILON_START,
                                     epsilon_end=ps.EPSILON_END,
                                     burn_in=ps.BURN_IN,
                                     frame_counter_increment=ps.FRAME_COUNTER_INC_STEP,
                                     prng=prng)
    while 1:
        main_controller.do()


if __name__ == '__main__':
    main()
