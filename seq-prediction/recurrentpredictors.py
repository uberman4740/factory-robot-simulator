import sys
import os
import dataloader
import time

import numpy as np

from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, JZS1, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint

from datetime import datetime

def save_list(l, filepath):
    f = open(filepath, 'w')
    for item in l:
        f.write("%s\n" % item)
    f.close()


def main(counter):
    out_dir = datetime.now().strftime('%Y%m%d') + 'out_' + str(counter) + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    rng = np.random.RandomState()

    # load data
    sequence_length = 500
    n_train = 2400
    n_test = 100
    n_data = sequence_length * (n_train + n_test)
    input_length = 21
    percept_length = 18

    print 'load data ({0})...'.format(n_data)
    data = dataloader.get_data(0, n_data + 1)
    inputs, percepts = dataloader.make_batches(data, sequence_length, crop_end=2)
    x_train = inputs[:n_train]
    y_train = percepts[:n_train]

    x_test = inputs[n_train:]
    y_test = percepts[n_train:]

    n_hidden = 180
    dropout = False
    early_stopping_patience = 5
    n_additional = 180

    #
    # n_hidden = rng.choice([20, 40, 80, 160, 320])
    # dropout = rng.choice([True, False], p=[0.2, 0.8])
    # early_stopping_patience = rng.choice([10, 20, 50])
    #
    # n_additional = rng.choice([0, 40, 60], p=[0.6, 0.2, 0.2])

    print 'build model...'
    model = Sequential()
    model.add(LSTM(input_dim=input_length,
                   output_dim=n_hidden,
                   activation='tanh',
                   inner_activation='hard_sigmoid',
                   init='glorot_uniform',
                   inner_init='orthogonal',
                   # forget_bias_init='one',
                   return_sequences=True))

    if dropout:
        model.add(Dropout(0.5))

    if n_additional > 0:
        model.add(LSTM(output_dim=n_additional,
                       activation='tanh',
                       inner_activation='hard_sigmoid',
                       init='glorot_uniform',
                       inner_init='orthogonal',
                       # forget_bias_init='one',
                       return_sequences=True))
        # model.add(GRU(output_dim=n_additional,
        #        activation='tanh',
        #        inner_activation='hard_sigmoid',
        #        init='glorot_uniform',
        #        inner_init='orthogonal',
        #        # forget_bias_init='one',
        #        return_sequences=True))
    if dropout:
        model.add(Dropout(0.5))

    model.add(GRU(output_dim=percept_length,
                   activation='sigmoid',
                   inner_activation='hard_sigmoid',
                   init='glorot_uniform',
                   inner_init='orthogonal',
                   # forget_bias_init='one',
                   return_sequences=True))

    tic = time.time()
    model.compile(loss='mean_absolute_error', optimizer='adam')
    compile_time = time.time() - tic
    print 'Compile time: {0} sec'.format(compile_time)

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
    model_checkpoint = ModelCheckpoint(out_dir + 'model_checkpoint.h5',
                                       monitor='val_loss',
                                       save_best_only=True)

    print 'start training...'

    tic = time.time()
    model.fit(x_train, y_train,
              batch_size=1,
              nb_epoch=400,
              validation_split=0.1,
              callbacks=[early_stopping, model_checkpoint],
              shuffle=False)
    training_duration = time.time() - tic

    score = model.evaluate(x_test, y_test, batch_size=1)

    save_list([n_hidden, dropout, early_stopping_patience, n_additional, compile_time, training_duration, score],
              out_dir + '_choice_and_result.dat')

    json_string = model.to_json()
    save_list([json_string],
              out_dir + 'json.dat')
    model.save_weights(out_dir + 'weights.h5')




    # --- Hyperparameter choices ---
    # input_layer_size_choices = [20, 50, 100, 200]
    # input_layer_type_choices = ['dense', 'lstm', 'jzs1']
    #
    # hidden_layer_count_choices = [0, 1]
    # hidden_layer_size_choices = [[20, 50, 100, 200],
    #                              [20, 50, 100, 200]]
    #
    # hidden_layer_type_choices = [['dense', 'lstm', 'jzs1'],
    #                              ['dense', 'lstm', 'jzs1']]
    #
    # output_layer_type_choices = ['dense', 'lstm', 'jzs1']
    #
    # l2_choices = [0.0, 1e-8, 1e-7, 1e-6]
    # dropout_choices = [False, True]
    # optimizer_choices = ['adam', 'rmsprop']
    #
    # init_choices = ['glorot_uniform', 'glorot_normal']
    #
    #
    # # --- Select Hyperparameters ---
    # input_layer_size = rng.choice(input_layer_size_choices)
    # input_layer_type = rng.choice(input_layer_type_choices)
    #
    # hidden_layer_count = rng.choice(hidden_layer_count_choices)
    # hidden_layer_sizes = [[0]] * hidden_layer_count
    # hidden_layer_types = [[None]] * hidden_layer_count
    #
    # for i in xrange(hidden_layer_sizes):
    #     hidden_layer_sizes[i] = rng.choice(hidden_layer_size_choices[i])
    #     hidden_layer_types[i] = rng.choice(hidden_layer_type_choices[i])
    #
    # output_layer_type = rng.choice(output_layer_type_choices)
    #
    # l2 = rng.choice(l2_choices)
    # dropout = rng.choice(dropout_choices)
    # optimizer = rng.choice(optimizer_choices)
    #
    # init = rng.choice(init_choices)
    #
    # # save hyperparameters
    # all_hyperparams = [
    #     input_layer_type,
    #     input_layer_size,
    #     hidden_layer_count,
    #     hidden_layer_sizes,
    #     hidden_layer_types,
    #     output_layer_type,
    #     l2,
    #     dropout,
    #     optimizer,
    #     init]
    #
    # save_list(all_hyperparams, out_dir + 'hyperparams.txt')
    #
    # # build model
    # model = Sequential()
    #
    # next_unit = LSTM(input_dim=input_length,
    #                  output_dim=input_layer_size,
    #                  activation='tanh',
    #                  inner_activation='hard_sigmoid',
    #                  init=init,
    #                  inner_init='orthogonal',
    #                  forget_bias_init='one',
    #                  return_sequences=True))
    #
    # for i in xrange(hidden_layer_count):
    #     if hidden_layer_types[i] == 'lstm':
    #         next_unit = LSTM(
    #                  output_dim=hidden_layer_sizes[i],
    #                  activation='tanh',
    #                  inner_activation='hard_sigmoid',
    #                  init=init,
    #                  inner_init='orthogonal',
    #                  forget_bias_init='one',
    #                  return_sequences=True))
    #
    #     elif hidden_layer_types[i] == 'jzs1':
    #         next_unit = LSTM(
    #                  output_dim=hidden_layer_sizes[i],
    #                  activation='tanh',
    #                  inner_activation='hard_sigmoid',
    #                  init=init,
    #                  inner_init='orthogonal',
    #                  forget_bias_init='one',
    #                  return_sequences=True))
    #
    #     else True:
    #         raise Exception('Unknown layer type')




    # train model w/ early stopping


    # save model architecture

    # save model weights

    # determine model accuracy


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise Exception('Need to specify counter!')

    main(int(sys.argv[1]))
