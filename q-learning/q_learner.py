#!/usr/bin/env python
"""
Q-Learner Module.

General nomenclature:
    percept: the agent's sensor information at a single point in time. Different 
        from "state" since the environment may be only partially observable.
    state short term memory (state_stm): number of consecutive percepts which are
        treated as a proxy for a state. 
    observation: a tuple (percept, action, reward), where action is the action
        taken in the previous time step, and reward is the reward collected since
        the last time step.
    minibatch (mb). Number of training instances used to determine the gradient
        of the q-function.  

"""

import os, sys

# Third-party imports
import numpy as np
import theano
import theano.tensor as T
sys.path.append('../labeled-experiments/nn-classifiers/')
from labeling_network import FullyConnectedLayer
import cPickle

__author__ = "Alexander Neitz"


floatX = theano.config.floatX

class ExpStoreInsufficientException(Exception):
    pass

class QLearner(object):
    """Manages observations and offers functionality to train a Q-function.
    
    """

    def __init__(self, q_function, exp_store_size, percept_length, n_actions,
                 state_stm, gamma, minibatch_size, prng):
        """
        Args:
            q_function (QFunction): working Q-function with correct number of inputs 
                and outputs
            exp_store_size (int): number of observations which are stored in learner's
                memory and which are used to train the Q-function
            percept_length (int): number of float values representing a single percept
            n_actions (int): number of actions the learner can perform. Actions are
                referred to by an integer index, starting at 0.
            state_stm (int): number of consecutive percepts which are used in a single
                training instance
            gamma (float): future reward discount per time step
            minibatch_size (int): number of training instances per training step
            prng (RandomState): numpy random number generator

        """
        assert q_function.n_out == n_actions
        assert q_function.n_in == percept_length * state_stm

        self.q_function = q_function
        self.state_stm = state_stm
        self.exp_store_size = exp_store_size
        self.percept_length = percept_length
        # self.n_actions = n_actions
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        self.prng = prng

        self.exp_store_percepts = np.zeros((exp_store_size, percept_length),
                                           dtype=floatX)
        self.exp_store_rewards = np.zeros((exp_store_size,), dtype=floatX)
        self.exp_store_actions = np.zeros((exp_store_size,), dtype='int32')
        self.exp_counter = 0
        self.exp_store_current_size = 0

    def add_observation(self, percept, previous_action, reward):
        """Adds an observation to the learners experience store.

        Args:
            percept (1D numpy floatX-array): the sensor information at the current 
                time step.
            previous_action (int): ID of the action taken at the previous time step.
            reward (float): collected reward since the last time step.

        """
        self.exp_store_percepts[self.exp_counter] = percept
        if self.exp_store_current_size > 0:
            self.exp_store_actions[(self.exp_counter - 1) % self.exp_store_size] = previous_action
            self.exp_store_rewards[(self.exp_counter - 1) % self.exp_store_size] = reward
        self.increment_exp_counter()

    def train_q_function(self, learning_rate):
        """Trains the Q-function with a minibatch from the learner's experience store.

        Args:
            learning_rate (float):
                Amount of the weight adjustment.

        """
        minibatch = self.assemble_minibatch()
        if minibatch is None:
            return -1.0
        percepts_before, percepts_after, actions, rewards = minibatch
        target_values = rewards + self.gamma*self.get_best_qs(percepts_after)
        return self.q_function.train(percepts_before, actions, target_values, learning_rate)

    def get_current_qs(self):
        """Returns q-values for the state which is described by the most recent observations."""
        if self.exp_counter >= self.state_stm:
            current_state = self.exp_store_percepts[self.exp_counter-self.state_stm:
                                                    self.exp_counter].flatten()
        elif self.exp_store_current_size == self.exp_store_size:
            current_state = np.append(self.exp_store_percepts[self.exp_counter-self.state_stm:],
                                      self.exp_store_percepts[:self.exp_counter])
#            print self.exp_counter
#            print self.state_stm
#            print current_state.shape
        else:
            # not enough percepts to construct state
            raise ExpStoreInsufficientException('Only %d percepts in experience store' % (self.exp_store_current_size))
            
        return self.q_function.get_q_values(current_state)

    def get_current_best_action(self):
        return np.argmax(self.get_current_qs())

    def get_current_best_q(self):
        return np.max(self.get_current_qs())

    def get_best_qs(self, percepts):
        qs = self.q_function.get_q_values_mb(percepts)
        return np.max(qs, axis=1)
    
    
#    def get_best_action(self, percepts):
#        qs = self.q_function.get_q_values

    def assemble_minibatch(self):
        """Constructs a minibatch of training examples from the experience store.

        Returns:
            4-tuple: 
            (percept-minibatch before, 
             percept-minibatch after, 
             action-minibatch, 
             reward-minibatch)
        """
        upper_index_bound = self.exp_store_current_size - self.state_stm 
        if upper_index_bound < 1:
            return None
        indices = self.prng.randint(0, upper_index_bound, self.minibatch_size)
        aug_indices = np.asarray([indices + i 
                                  for i in range(self.state_stm)]).T
        return (self.get_percept_minibatch(aug_indices),
                self.get_percept_minibatch(aug_indices + 1),
                self.get_action_minibatch(aug_indices[:, -1]),
                self.get_reward_minibatch(aug_indices[:, -1]))

    def get_percept_minibatch(self, aug_indices):
        """Constructs a minibatch of percepts

        Args:
            aug_indices (2D numpy int-array): Matrix of indices which specify 
                positions in percept experience store. Rows correspond to the training
                instances in the minibatch, columns correspond to the different 
                percepts that constitute the training example.  

        Returns:
            Percept-minibatch as 2D-numpy-array (floatX).
        """
        return self.exp_store_percepts[aug_indices] \
               .reshape(self.minibatch_size,
                        self.state_stm * self.percept_length)

    def get_action_minibatch(self, indices):
        """Constructs a minibatch of actions.

        Args:
            indices (1D numpy int-array): Vector of indices which specify positions
            in the action experience store for each example in the minibatch.
        
        Returns:
            Action-minibatch as 1D-numpy-array (int).
        """
        return self.exp_store_actions[indices]

    def get_reward_minibatch(self, indices):
        """Constructs a minibatch of rewards.

        Args:
            indices (1D numpy int-array): Vector of indices which specify positions
            in the reward experience store for each example in the minibatch.

        Returns:
            Reward-minibatch as 1D-numpy-array (floatX)
        """
        return self.exp_store_rewards[indices]

    def increment_exp_counter(self):
        """Increments experience counter (wraps around) and keeps track of its size."""
        self.exp_counter = (self.exp_counter + 1) % self.exp_store_size
        if self.exp_store_current_size < self.exp_store_size:
            self.exp_store_current_size += 1


class QNetwork(object):
    """Represents Q-Function which can be used in the Q-Learner.
    """

    def __init__(self, layers, minibatch_size):
        """
        Args:
            layers: list of layer objects
            minibatch_size (int): number of examples per minibatch
        """
        self.layers = layers
        self.minibatch_size = minibatch_size
        self.inpt = T.matrix('inpt')
        self.single_inpt = T.vector('single_inpt')
        self.target_qs = T.vector('target_qs')
        self.actions = T.ivector('actions')

        init_layer = self.layers[0]
        init_layer.set_inpt(self.inpt, self.inpt, self.minibatch_size)
        init_layer.set_single_inpt(self.single_inpt)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.minibatch_size)
            layer.set_single_inpt(
                prev_layer.single_output)

        self.n_out = layers[-1].n_out
        self.n_in = layers[0].n_in
        self.output = layers[-1].output
        self.single_output = self.layers[-1].single_output

        learning_rate = T.scalar('learning_rate')
        params = [param for layer in self.layers for param in layer.params]
        cost = self.get_cost()
        grads = T.grad(cost, params)
        updates = [(param, param - learning_rate*grad)
                   for param, grad in zip(params, grads)]
        self.train = theano.function([self.inpt,
                                      self.actions,
                                      self.target_qs,
                                      learning_rate],
                                     cost,
                                     updates=updates)
        self.output_fn = theano.function([self.inpt], self.output)
        self.single_output_fn = theano.function([self.single_inpt], self.single_output)

    def get_cost(self):
        """Returns symbolic squared difference between targets and relevant q-values"""
        return T.mean((self.target_qs - \
            self.output[T.arange(T.shape(self.actions)[0]), self.actions]) ** 2)

    def get_q_values_mb(self, state_mb):
        """Returns current estimation of q-values for the state-minibatch"""
        return self.output_fn(state_mb)

    def get_q_values(self, state):
        return self.single_output_fn(state)

    def save_as_file(self, filename_prefix):
        # save all layers
        for i, layer in enumerate(self.layers):
            filename = filename_prefix + '_layer' + str(i) + '.save'
            try:
                os.remove(filename)
            except OSError:
                print 'Creating file', filename
            
            f = file(filename, 'wb')
            cPickle.dump(layer, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
        i = len(self.layers)
        while os.path.isfile(filename + '_layer' + str(i) + '.save'):
            os.remove(filename + '_layer' + str(i) + '.save')
            i += 1


    @classmethod
    def load_from_file(cls, filename, mini_batch_size):
        # load all layers
        layers = []
        i = 0
        prefix = filename + '_layer'
        while os.path.isfile(prefix + str(i) + '.save'):
            f = file(prefix + str(i) + '.save', 'rb')
            layers.append(cPickle.load(f))
            f.close()
            i += 1
        
        if len(layers) == 0:
            print 'Network not found!'
            return None
        
        print 'Loading network:', len(layers), 'layers loaded.'
        return cls(layers, mini_batch_size)


