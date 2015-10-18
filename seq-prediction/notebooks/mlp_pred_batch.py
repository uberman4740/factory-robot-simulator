mpl_cols = ['#3388dd', '#aa3377', '#449911']

import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(
        os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],
                     '..', '..', 'labeled-experiments', 'nn-classifiers')))
print 'Added {0} to path.'.format(cmd_folder)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import labeling_network as lbln
from matplotlib import pyplot as plt
import theano
import theano.tensor as T
import numpy as np
from labeling_network import FullyConnectedLayer, ConvPoolLayer

import time
import datetime

rng = np.random.RandomState(12345678)

dataPath = '../../../factory-robot-data/imgs_2015-10-17/'


def load_action_data(filename, lower, upper):
    action_file = open(filename)
    lines = action_file.readlines()[lower:upper]
    action_file.close()
    data = np.asarray([int(l) for l in lines])
    return data

def convert_actions(actions, n_actions):
    result = []
    for a in actions:
        x = [0.0] * n_actions
        x[a] = 1.0
        result.append(x)
    return np.asarray(result, dtype=theano.config.floatX)



def construct_training_examples(label_data, action_data, delta_t, n_past, n_future):
    assert len(label_data) == len(action_data)
    percept_len = label_data.shape[1]
    n_actions = action_data.shape[1]

    xs_length = n_past*percept_len + (n_past + n_future - 1)*n_actions
    t_length = n_future*percept_len

    training_data_xs = np.empty((len(label_data) - (n_past + n_future - 1)*delta_t, xs_length),
                                dtype=theano.config.floatX)
    training_data_ts = np.empty((len(label_data) - (n_past + n_future - 1)*delta_t, t_length),
                                dtype=theano.config.floatX)

    for i in xrange(n_past * delta_t, len(label_data) - ((n_future-1)*delta_t + 1)):
        example = []
        for j in xrange(n_future):
            training_data_ts[i - n_past*delta_t, j*percept_len: (j+1)*percept_len] = np.asarray(
                label_data[i+j], dtype=theano.config.floatX)

            if (n_future - j) > 1:
                xs_a = np.mean(action_data[i + j*delta_t:
                                           i + (j+1)*delta_t] , axis=0)
#                 print -(n_future-j-1)*n_actions
#                 print -(n_future-j-2)*n_actions
#                 print xs_a.shape
#                 print
                if n_future - j == 2:
                    training_data_xs[i - n_past*delta_t, -(n_future-j-1)*n_actions:] = xs_a
                else:
                    training_data_xs[i - n_past*delta_t, -(n_future-j-1)*n_actions:
                                                         -(n_future-j-2)*n_actions] = xs_a
        for j in xrange(n_past):
            xs_d = label_data[i - (n_past*delta_t) + j*delta_t]
            xs_a = np.mean(action_data[i - (n_past*delta_t) + j*delta_t :
                                       i - (n_past*delta_t) + (j+1)*delta_t] , axis=0)
            training_data_xs[i - n_past*delta_t, j*(percept_len + n_actions):
                                                 (j+1)*percept_len + j*n_actions] = xs_d
            training_data_xs[i - n_past*delta_t, (j+1)*percept_len + j*n_actions:
                                                 (j+1)*(percept_len + n_actions)] = xs_a

    return training_data_xs, training_data_ts


def shuffle_data(data, rng):
    xs, ts = data
    index_set = np.asarray(range(len(xs)))
    rng.shuffle(index_set)
    return xs[index_set], ts[index_set]


n_train = 70000
n_valid = 5000
n_test = 1000

n_direction_sensors=7
n_classes=2
n_actions=3
delta_t = 2
n_past = 12
n_future = 5

lower = 0
upper = n_train + n_valid + n_test + delta_t*n_past



load_time_start = time.time()
label_data_raw = lbln.load_labeling_data(dataPath+'labels.dat', lower, upper, mask=-1,
                                         n_direction_sensors=n_direction_sensors,
                                         n_classes=n_classes)
actions_raw = convert_actions(load_action_data(dataPath+'actions.dat', lower, upper),
                              n_actions=n_actions)

all_data = construct_training_examples(label_data_raw, actions_raw, delta_t, n_past, n_future)
all_data = shuffle_data(all_data, rng)

training_xs = theano.shared(all_data[0][:n_train], borrow=True)
training_ts = theano.shared(all_data[1][:n_train], borrow=True)

valid_xs = theano.shared(all_data[0][n_train: n_train+n_valid], borrow=True)
valid_ts = theano.shared(all_data[1][n_train: n_train+n_valid], borrow=True)

test_xs = theano.shared(all_data[0][n_train+n_valid:], borrow=True)
test_ts = theano.shared(all_data[1][n_train+n_valid:], borrow=True)


print 'Loading data took {0:.5} seconds'.format(time.time() - load_time_start)


counter = 0
n_epochs = 100
while 1:

    savedir = 'trained-networks/' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # param choice
    mini_batch_sizes = [5, 10, 20]
    rms1s = [0.0001, 0.001, 0.01]
    rms2s = [0.95, 0.9, 0.8]
    rms4s = [0.5, 1.0, 5.0]
    n_hiddens = [300, 500, 1000, 1500]
    n_hiddens_2 = [200, 300]
    p_multi = 0.2


    # choose params
    mini_batch_size = rng.choice(mini_batch_sizes)
    rms1 = rng.choice(rms1s)
    rms2 = rng.choice(rms2s)
    rms4 = rng.choice(rms4s)
    n_hidden = rng.choice(n_hiddens)
    n_hidden_2 = rng.choice(n_hiddens_2)
    two_layers = rng.binomial(1, p_multi)


    # save params
    paramfile = open(savedir + 'params.txt', 'w')
    paramfile.write(str(mini_batch_size) + '\n')
    paramfile.write(str(rms1) + '\n')
    paramfile.write(str(rms2) + '\n')
    paramfile.write(str(rms4) + '\n')
    paramfile.write(str(n_hidden) + '\n')
    paramfile.write(str(n_hidden_2) + '\n')
    paramfile.write(str(two_layers) + '\n')
    paramfile.close()

    if two_layers:
        network = lbln.Network([
                FullyConnectedLayer(n_in=n_past*(n_direction_sensors*n_classes + n_actions) + (n_future-1)*n_actions,
                                    n_out=n_hidden),
                FullyConnectedLayer(n_in=n_hidden,
                                    n_out=n_hidden_2),
                FullyConnectedLayer(n_in=n_hidden_2,
                                    n_out=n_direction_sensors*n_classes*n_future)
            ], mini_batch_size)
    else:
                network = lbln.Network([
                FullyConnectedLayer(n_in=n_past*(n_direction_sensors*n_classes + n_actions) + (n_future-1)*n_actions,
                                    n_out=n_hidden),
                FullyConnectedLayer(n_in=n_hidden,
                                    n_out=n_direction_sensors*n_classes*n_future)
            ], mini_batch_size)


    network.SGD((training_xs, training_ts),
                n_epochs,
                mini_batch_size,
                0.2,
                (valid_xs, valid_ts),
                (test_xs, test_ts),
                savedir + 't_nf' + str(n_future) + '_np' + str(n_past) + '_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
                learning_curve_file_name=savedir + 'decoder_learning_curve_bigdata',
                rmsprop=(rms1, rms2, 1e-6, rms4)
               )
    counter += 1