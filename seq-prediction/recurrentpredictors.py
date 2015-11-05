import sys
import os
import dataloader
import time

import numpy as np

import theano.tensor as T
import theano

from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, JZS1, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop

from datetime import datetime


def save_list(l, filepath):
    f = open(filepath, 'w')
    for item in l:
        f.write("%s\n" % item)
    f.close()


def sigma_mu_loss(percept_length, sigma_min, y_true, y_pred):
    sigmas_pred = sigma_min + y_pred[:, percept_length:]
    mus_pred = y_pred[:, :percept_length]
    return T.mean(((y_true - mus_pred) ** 2) / (2 * sigmas_pred ** 2) + T.log(sigmas_pred))


def laplace_loss(percept_length, sigma_min, y_true, y_pred):
    sigmas_pred = sigma_min + y_pred[:, percept_length:]
    mus_pred = y_pred[:, :percept_length]
    return T.mean((T.abs_(y_true - mus_pred)) / sigmas_pred + T.log(sigmas_pred))


def main(counter, prediction_interval, loss_function, data_root_path):

    print 'prediction_interval:', prediction_interval
    print 'loss_function:', loss_function
    print 'data:', data_root_path


    out_dir = 'results/' + datetime.now().strftime('%Y%m%d%H%M%S') + 'p_' + str(prediction_interval) + 'out_' + str(
        counter) + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    rng = np.random.RandomState()

    # load data
    sequence_length = 500
    n_train = 2400
    n_test = 100
    n_data = sequence_length * (n_train + n_test)
    input_length = 18 + 3 * prediction_interval
    percept_length = 18
    sigma_min = 1e-4
    n_epochs = 400

    print 'load data ({0})...'.format(n_data)
    data = dataloader.get_data(0, n_data + 1,
                               n_actions=3,
                               prediction_interval=prediction_interval,
                               data_path_lab=data_root_path+'labels.dat',
                               data_path_act=data_root_path+'actions.dat')
    inputs, percepts = dataloader.make_batches(data,
                                               sequence_length,
                                               n_actions=3,
                                               prediction_interval=prediction_interval,
                                               crop_end=2)
    x_train = inputs[:n_train]
    y_train = percepts[:n_train]

    x_test = inputs[n_train:]
    y_test = percepts[n_train:]

    # n_hidden = rng.choice([100, 200, 500])
    n_hidden = rng.randint(500, 800)
    early_stopping_patience = 3
    n_additional = rng.randint(800, 1000)
    # n_additional = rng.choice([0, 100, 200])
    # n_additional = 0

    print 'build model...'
    model = Sequential()
    model.add(LSTM(n_hidden,
                   input_dim=input_length,
                   return_sequences=True))

    if n_additional > 0:
        model.add(LSTM(n_additional,
                       return_sequences=True))

    # model.add(Activation('tanh'))
    # model.add(Dropout(p=0.5))
    output_length = 2*percept_length if loss_function in ('gauss', 'laplace') else percept_length
    model.add(TimeDistributedDense(output_length))
    model.add(Activation('sigmoid'))


    # model = Sequential()
    # model.add(TimeDistributedDense(n_hidden,
    #                input_dim=input_length))
    # model.add(Activation('tanh'))
    # # model.add(Dropout(p=0.5))
    # output_length = 2*percept_length if loss_function in ('gauss', 'laplace') else percept_length
    # model.add(TimeDistributedDense(output_length))
    # model.add(Activation('sigmoid'))


    tic = time.time()

    def sigma_mu_loss_(y_true, y_pred):
        return sigma_mu_loss(percept_length, sigma_min, y_true, y_pred)

    def laplace_loss_(y_true, y_pred):
        return laplace_loss(percept_length, sigma_min, y_true, y_pred)

    if loss_function == 'gauss':
        loss = sigma_mu_loss_
    elif loss_function == 'laplace':
        loss = laplace_loss_
    else:
        loss = loss_function

    # optimizer = SGD(lr=0.01, momentum=0.8, nesterov=True, clipnorm=5.0)
    optimizer = Adam(clipnorm=5.0)
    model.compile(loss=loss, optimizer=optimizer)
    compile_time = time.time() - tic
    print 'Compile time: {0} sec'.format(compile_time)

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
    model_checkpoint = ModelCheckpoint(out_dir + 'model_checkpoint.h5',
                                       monitor='val_loss',
                                       save_best_only=True)

    choice_list = [prediction_interval, n_hidden, early_stopping_patience, n_additional, compile_time]
    print 'choices:', choice_list
    save_list(choice_list,
              out_dir + '_choice.dat')
    json_string = model.to_json()
    save_list([json_string],
              out_dir + 'architecture.json')


    print 'start training...'

    tic = time.time()
    model.fit(x_train, y_train,
              batch_size=1,
              nb_epoch=n_epochs,
              validation_split=0.1,
              callbacks=[early_stopping, model_checkpoint],
              shuffle=False)
    training_duration = time.time() - tic

    score = model.evaluate(x_test, y_test, batch_size=1)

    save_list([training_duration, score],
              out_dir + 'result.dat')


    model.save_weights(out_dir + 'weights.h5', overwrite=True)


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise Exception('3 Command line arguments needed: counter, '
                        'prediction_interval, loss_fn, data_path')

    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
