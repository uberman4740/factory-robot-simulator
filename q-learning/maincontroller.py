#!/usr/bin/env python

"""maincontroller.py: Central management of Q-Learning data flow and communication."""

import socket
import time
import threading
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from q_learner import QLearner, QNetwork
import inputlistener
from sensordecoder import SensorDecoder
import params as ps
from labeling_network import FullyConnectedLayer, linear, Network
from log import log
from expstoreprint import export_exp_store
# import livecharting
import pyqtlivecharting
import sys

from qualitycontrol import QualityLogger


# import livebarchart

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
                 prng=None,
                 training_error_smoothing=0.0,
                 log_path=None,
                 log_write_period=1000,
                 reward_smoothing=0.0,
                 quality_logger=None):
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
        self.smooth_training_error = 0.0
        self.training_error_smoothing = training_error_smoothing
        self.log_path = log_path
        self.log_write_period = log_write_period
        self.rewards_log = []
        self.smooth_reward = 0.0
        self.reward_smoothing = reward_smoothing
        self.quality_logger = quality_logger

    def do(self):
        """Check for data receipt and send decisions to remote host. """
        if self.current_state == MainController.RECEIVE:
            if self.timeout():
                log('Timeout.', 1)
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
                        differential_reward = total_reward - self.current_total_reward
                        self.remember_percept(percept,
                                              prev_action,
                                              differential_reward)
                        self.current_total_reward = total_reward
                        self.rewards_log.append(differential_reward)
                        if (self.total_steps % self.log_write_period) == 0:
                            append_to_log(self.log_path + 'rewards.log', self.rewards_log)
                            self.rewards_log = []
                        self.smooth_reward = (self.reward_smoothing * self.smooth_reward +
                                              (1 - self.reward_smoothing) * differential_reward)
                        if smooth_reward_charting is not None:
                            smooth_reward_charting.set_current_value(0, 30.0*self.smooth_reward)

                        self.current_decision = self.get_decision()
                        self.advance_frame()
                        self.current_state = MainController.SEND

                        # TODO: REMOVE
                        # if self.total_steps == 30:
                        #     export_exp_store(self.q_learner.exp_store_percepts,
                        #                      self.q_learner.exp_store_actions,
                        #                      self.q_learner.exp_store_rewards,
                        #                      filepath='exp-store-logs/',
                        #                      state_stm=self.q_learner.state_stm,
                        #                      lower=10,
                        #                      upper=20)
                        #     raise SystemExit

        if self.current_state == MainController.SEND:
            log('Sending... (total_steps={0}, current_decision={1})'.format(self.total_steps,
                                                                            self.current_decision))
            self.send_decision(self.current_decision)
            self.current_state = MainController.LEARN
            self.quality_control()

        if self.current_state == MainController.LEARN:
            if self.total_steps > self.burn_in:
                mean_error = 0.0
                for i in xrange(self.learning_iterations_per_step):
                    err = self.q_learner.train_q_function(self.learning_rate)
                    mean_error = 1. * i * mean_error / (i + 1) + 1. * err / (i + 1)
                self.smooth_training_error = ((1 - self.training_error_smoothing) * mean_error
                                              + self.training_error_smoothing *
                                              self.smooth_training_error)

                current_qs = self.q_learner.get_current_qs()
                # log('qs: {0}'.format(current_qs), 1)
                if q_charting is not None:
                    for i, q in enumerate(current_qs):
                        q_charting.set_current_value(i, q)
                if error_charting is not None:
                    error_charting.set_current_value(0, self.smooth_training_error)
                if (self.total_steps % self.log_write_period) == 0:
                    self.q_learner.q_function.save_as_file(self.log_path + 'q_function')
            self.current_state = MainController.RECEIVE

    def quality_control(self):
        self.quality_logger.set_value(self.current_total_reward, self.total_steps)
        sigma = self.quality_logger.get_sigma()
        if sigma is not None:
            print 'sigma: {0}'.format(sigma)

    def remember_percept(self, percept, last_action, previous_reward):
        """Encode raw percept and pass encoding, last action and reward to Q-Learner."""
        encoding = self.state_encoder_fn(percept)
        if encoding_charting is not None:
            encoding_charting.set_values(0, encoding[5:10])
        self.q_learner.add_observation(encoding.astype(np.float32),
                                       last_action,
                                       previous_reward)
        # self.q_learner.add_observation(percept,
        #                                last_action,
        #                                previous_reward)

    def get_decision(self):
        """Use Q-Function to decide for current action in epsilon-greedy way."""
        epsilon = max(self.epsilon_end,
                      self.epsilon_start -
                      (self.epsilon_start - self.epsilon_end) *
                      1. * self.total_steps / self.epsilon_decrease_duration)
        randaction_p = 0 if epsilon == 0 else 1. / (self.randaction_duration * (1. / epsilon - 1.) + 1.)
        log('epsilon: {0:.4}'.format(epsilon), 1)
        if self.current_randaction_termination > self.total_steps:
            return self.frame_counter, self.current_randaction
        if self.prng.uniform(0, 1) >= randaction_p \
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


def load_labeling_function(filename, mb_size, use_layer=None):
    net = Network.load_from_file(filename, mb_size)
    if use_layer is not None:
        net.single_output = net.layers[use_layer].single_output
        net.output = net.layers[use_layer].output
    return net


def load_q_network(filename, state_stm, percept_length,
                   q_hidden_neurons, n_actions, mb_size):
    print 'load_q_network'
    if filename is not None:
        print 'return load_from_file'
        return QNetwork.load_from_file(filename, mb_size)
    else:
        hidden_layer = FullyConnectedLayer(state_stm * percept_length,
                                           q_hidden_neurons)
        output_layer = FullyConnectedLayer(q_hidden_neurons,
                                           n_actions,
                                           activation_fn=linear)
        print 'return new QNetwork.'
        return QNetwork([hidden_layer, output_layer], minibatch_size=mb_size)


def append_to_log(filepath, values):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'a') as file:
        for value in values:
            file.write(str(value) + '\n')


def copy_parameter_file(path):
    if not os.path.exists(path):
        os.makedirs(path)
    shutil.copy2('params.py', path)


def main():
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
                                          ps.MB_SIZE,
                                          ps.LABELING_NETWORK_USE_LAYER)
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

    log_path = ps.LOG_PATH + time.strftime('%Y-%m-%d_%H-%M-%S') + '/'
    copy_parameter_file(log_path)

    # quality_logger = QualityLogger(ps.QUALITY_LOG_PATH)

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
                                     prng=prng,
                                     training_error_smoothing=ps.TRAIN_ERROR_SMOOTHING,
                                     log_path=log_path,
                                     reward_smoothing=ps.REWARD_SMOOTHING,
                                     quality_logger=QualityLogger(ps.QUALITY_LOG_PATH))


    print 'Starting main loop.'
    while 1:
        main_controller.do()


if __name__ == '__main__':
    q_charting = None
    error_charting = None
    encoding_charting = None
    smooth_reward_charting = None


    anim_thread = threading.Thread(target=main)
    anim_thread.daemon = True
    anim_thread.start()

    time.sleep(2)
    print 'start plotting'
    q_charting = pyqtlivecharting.LiveChartingLines(n_curves=ps.N_ACTIONS,
                                               y_min=-1.0,
                                               y_max=5.0,
                                               curve_width=4,
                                               steps=300,
                                               title='Q-values',
                                               ylabel='q-value')
    # error_charting = pyqtlivecharting.LiveChartingLines(n_curves=1,
    #                                                y_min=0,
    #                                                y_max=2.0,
    #                                                curve_width=2,
    #                                                steps=600,
    #                                                title='Training error',
    #                                                ylabel='mean error')
    # encoding_charting = pyqtlivecharting.LiveChartingLines(n_curves=1,
    #                                                y_min=0,
    #                                                y_max=1.0,
    #                                                curve_width=5,
    #                                                steps=5,
    #                                                title='Encoding',
    #                                                ylabel='signal')
    smooth_reward_charting = pyqtlivecharting.LiveChartingLines(n_curves=1,
                                                                y_min=-1.0,
                                                                y_max=1.0,
                                                                curve_width=3,
                                                                steps=5000,
                                                                title='Reward',
                                                                ylabel='reward')
    pyqtlivecharting.LiveChartingLines.run()

    # livecharting.LiveCharting(n_curves=ps.N_ACTIONS,
    #                           ymin=-1.0,
    #                           ymax=3.5,
    #                           ylabel='Q-values',
    #                           data_labels=ps.ACTION_NAMES,
    #                           x_resolution=50)
    # plt.show()
    # time.sleep(3600000)
