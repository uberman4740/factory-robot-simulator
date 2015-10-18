"""labeling_network.py
~~~~~~~~~~~~~~

Based on network3.py by Michael Nielsen

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

"""

#### Libraries
# Standard library
import cPickle
import gzip
import os

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample
import scipy.misc
import time

# Activation functions for neurons
def linear(z): return z


def ReLU(z): return T.maximum(0.0, z)


from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = False
if GPU:
    print "Trying to run under a GPU.  If this is not desired, then modify " + \
          "network3.py\nto set the GPU flag to False."
    try:
        theano.config.device = 'gpu'
    except:
        pass  # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU.  If this is not desired, then the modify " + \
          "network3.py to set\nthe GPU flag to True."


def load_labeling_data(filename, lower, upper, mask=-1, n_direction_sensors=5, n_classes=5):
    labels = open(filename)
    lines = labels.readlines()[lower:upper]
    labels.close()
    data = np.asarray([[float(d) for d in l.split(',')[:-1]] for l in lines])

    if not mask == -1:
        data = data[:, n_direction_sensors * mask: n_direction_sensors * (mask + 1)]
        return data.reshape(upper - lower, n_direction_sensors)
    else:
        return data.reshape(upper - lower, n_direction_sensors * n_classes)


def load_training_img(index, file_path, file_prefix):
    return scipy.misc.imread(file_path + file_prefix + str(index).zfill(6) + '.png')[:, :, :-1]


def load_images(lower, upper, file_path, file_prefix):
    return np.asarray([load_training_img(i, file_path, file_prefix)
                       for i in range(lower, upper)])


def normalize_and_flatten(imgs):
    return (imgs / 255.0).reshape(imgs.shape[0], imgs.shape[1] * imgs.shape[2] * imgs.shape[3])


#### Load the Image data
def load_data_shared(file_path_images,
                     file_prefix_images,
                     filename_labels,
                     n_train=1000,
                     n_validation=300,
                     n_test=200,
                     label_mask=-1):
    training_images = normalize_and_flatten(load_images(0, n_train, file_path_images, file_prefix_images))
    validation_images = normalize_and_flatten(
        load_images(n_train, n_train + n_validation, file_path_images, file_prefix_images))
    test_images = normalize_and_flatten(
        load_images(n_train + n_validation, n_train + n_validation + n_test, file_path_images, file_prefix_images))

    training_labels = load_labeling_data(filename_labels, 0, n_train, label_mask)
    validation_labels = load_labeling_data(filename_labels, n_train, n_train + n_validation, label_mask)
    test_labels = load_labeling_data(filename_labels, n_train + n_validation, n_train + n_validation + n_test,
                                     label_mask)

    training_data = (training_images, training_labels)
    validation_data = (validation_images, validation_labels)
    test_data = (test_images, test_labels)

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, shared_y

    return [shared(training_data), shared(validation_data), shared(test_data)]


def RMSProp(cost, params, lr=0.001, rho=0.9, epsilon=1e-6, step_rate=1.0):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = step_rate * g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


#### Main class used to construct and train networks
class Network(object):
    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.matrix("y")
        self.x_single = T.vector("x_single")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        init_layer.set_single_inpt(self.x_single)

        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
            layer.set_single_inpt(
                prev_layer.single_output)

        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        self.single_output = self.layers[-1].single_output

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

    def get_single_output(self, input_to_classify):
        return self.single_output.eval({self.x_single: input_to_classify})

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, best_file_name=None, lmbda=0.0, learning_curve_file_name=None,
            rmsprop=None):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data) / mini_batch_size
        num_validation_batches = size(validation_data) / mini_batch_size
        num_test_batches = size(test_data) / mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + \
               0.5 * lmbda * l2_norm_squared / num_training_batches
        grads = T.grad(cost, self.params)
        if rmsprop is not None:
            updates = RMSProp(cost, self.params,
                              lr=rmsprop[0],
                              rho=rmsprop[1],
                              epsilon=rmsprop[2],
                              step_rate=rmsprop[3])
        else:
            updates = [(param, param - eta * grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar()  # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                    training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            },
            on_unused_input='warn')
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                    test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                    test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            },
            on_unused_input='warn')
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].output,
            givens={
                self.x:
                    test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })

        # Do the actual training
        best_validation_accuracy = float('-inf')
        training_set_costs = []
        for epoch in xrange(epochs):
            print('Epoch {0}: '.format(epoch))
            epoch_start_time = time.time()
            if training_set_costs:
                print '  training error:   {1}'.format(epoch, np.mean(training_set_costs))

            if learning_curve_file_name:
                with open(learning_curve_file_name + 'train_costs.lcurve', "a") as f:
                    f.write(str(np.mean(training_set_costs)) + '\n')

            training_set_costs = []
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches * epoch + minibatch_index
                # if iteration % 1000 == 0:
                #     print("--- Training mini-batch number {0}".format(iteration) + " ---")

                cost_ij = train_mb(minibatch_index)
                training_set_costs.append(cost_ij)

                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print('  validation error: {0:.5}'.format(validation_accuracy))
                    if learning_curve_file_name:
                        with open(learning_curve_file_name + 'validation_accuracies.lcurve', "a") as f:
                            f.write(str(-validation_accuracy) + '\n')

                    if validation_accuracy >= best_validation_accuracy:
                        if best_file_name:
                            self.save_as_file(best_file_name)

                        print('    (Best so far.)')
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print('  test error:        {0:.5}'.format(
                                -test_accuracy))
            print 'Epoch time: {0:.4}s'.format(time.time() - epoch_start_time)
            print '------------------'
        print("Finished training network.")
        print("Best validation accuracy of {0:.5} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.5}".format(test_accuracy))


#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        w_bound = np.sqrt(6.0 / (np.prod(filter_shape[1:] + n_out)))
        self.w = theano.shared(
            np.asarray(
                np.random.uniform(-w_bound, w_bound, size=filter_shape),
                # np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                # np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                np.zeros((filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def __getstate__(self):
        return (self.filter_shape,
                self.image_shape,
                self.poolsize,
                self.activation_fn,
                self.w.get_value(borrow=True),
                self.b.get_value(borrow=True))

    def __setstate__(self, state):
        self.filter_shape = state[0]
        self.image_shape = state[1]
        self.poolsize = state[2]
        self.activation_fn = state[3]

        self.w = theano.shared(np.asarray(state[4], dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(state[5], dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output  # no dropout in the convolutional layers

    def set_single_inpt(self, inpt):
        single_image_shape = (1, self.image_shape[1], self.image_shape[2], self.image_shape[3])
        self.single_inpt = inpt.reshape(single_image_shape)
        conv_out = conv.conv2d(
            input=self.single_inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=single_image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.single_output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                # np.random.normal(
                #     loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                np.random.uniform(
                    -np.sqrt(6.0 / (n_in + n_out)), np.sqrt(6.0 / (n_in + n_out)), size=(n_in, n_out)),

                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(
                # np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                np.zeros((n_out,)),
                dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def __getstate__(self):
        return (self.n_in,
                self.n_out,
                self.activation_fn,
                self.p_dropout,
                self.w.get_value(borrow=True),
                self.b.get_value(borrow=True))

    def __setstate__(self, state):
        self.n_in = state[0]
        self.n_out = state[1]
        self.activation_fn = state[2]
        self.p_dropout = state[3]
        self.w = theano.shared(np.asarray(state[4], dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.asarray(state[5], dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def set_single_inpt(self, inpt):
        self.single_inpt = inpt.reshape((self.n_in,))
        self.single_output = self.activation_fn(
            (1 - self.p_dropout) * T.dot(self.single_inpt, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."

        # return -T.mean(abs(self.output - y) / (1.01 - abs(self.output - y)))
        # return T.mean(T.log(1.0000001 - abs(self.output - y)))
        return -T.mean((self.output - y) ** 2)
        # return -T.mean(T.ones_like(y))

    def cost(self, net):
        "Return the cost."
        return T.mean((self.output_dropout - net.y) ** 2)
        # return T.mean(abs(self.output_dropout - net.y) / (1.01 - abs(self.output_dropout - net.y)))
        # return -T.mean(T.log(1.0000001 - abs(self.output_dropout - net.y)))
        # return -T.mean( T.log(1 - abs( ) ) )
        # return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])


class SparseLayer(object):
    #     def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
    def __init__(self, from_shape, to_shape, tiles, activation_fn=sigmoid):
        self.from_shape = from_shape
        self.to_shape = to_shape
        self.tiles = tiles
        n_from_shape = from_shape[0] * from_shape[1]
        n_to_shape = to_shape[0] * to_shape[1]
        n_tiles = tiles[0] * tiles[1]
        self.n_in = n_from_shape * n_tiles
        self.n_out = n_to_shape * n_tiles
        self.activation_fn = activation_fn

        # Initialize weights and biases
        self.ws = [theano.shared(
            np.asarray(
                np.random.uniform(
                    -np.sqrt(6.0 / (n_from_shape + n_to_shape)), np.sqrt(6.0 / (n_from_shape + n_to_shape)),
                    size=(n_from_shape, n_to_shape)),
                dtype=theano.config.floatX),
            name='w' + str(i) + ',' + str(j), borrow=True)
                   for i in xrange(tiles[0]) for j in xrange(tiles[1])]
        self.b = theano.shared(
            np.asarray(
                np.zeros((self.n_out,)),
                dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = self.ws + [self.b]

    #     def __getstate__(self):
    #         return (self.n_in,
    #                 self.n_out,
    #                 self.activation_fn,
    #                 self.p_dropout,
    #                 self.w.get_value(borrow=True),
    #                 self.b.get_value(borrow=True))

    #     def __setstate__(self, state):
    #         self.n_in = state[0]
    #         self.n_out = state[1]
    #         self.activation_fn = state[2]
    #         self.p_dropout = state[3]
    #         self.w = theano.shared(np.asarray(state[4], dtype=theano.config.floatX), borrow=True)
    #         self.b = theano.shared(np.asarray(state[5], dtype=theano.config.floatX), borrow=True)
    #         self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        inpt_height = self.from_shape[0] * self.tiles[0]
        inpt_width = self.from_shape[1] * self.tiles[1]
        output_height = self.to_shape[0] * self.tiles[0]
        output_width = self.to_shape[1] * self.tiles[1]

        self.inpt = inpt.reshape((mini_batch_size, inpt_height, inpt_width))
        shaped_output = T.zeros((mini_batch_size, output_height, output_width))
        for i in xrange(self.tiles[0]):
            for j in xrange(self.tiles[1]):
                inpt_tile = self.inpt[:,
                            i * self.from_shape[1]:(i + 1) * self.from_shape[1],
                            j * self.from_shape[0]:(j + 1) * self.from_shape[0]] \
                    .reshape((mini_batch_size, self.from_shape[0] * self.from_shape[1],))

                output_tile_flat = self.activation_fn(T.dot(inpt_tile, self.ws[i * self.tiles[1] + j]))
                output_tile = T.reshape(output_tile_flat, (mini_batch_size, self.to_shape[0], self.to_shape[1]))
                shaped_output = T.set_subtensor(shaped_output[:,
                                                i * self.to_shape[0]:(i + 1) * self.to_shape[0],
                                                j * self.to_shape[1]:(j + 1) * self.to_shape[1]],
                                                output_tile)

        self.output = T.reshape(shaped_output, (mini_batch_size, self.n_out)) + self.b
        #         self.inpt_dropout = dropout_layer(
        #             inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        #         self.output_dropout = self.activation_fn(
        #             T.dot(self.inpt_dropout, self.w) + self.b)
        self.output_dropout = self.output  # TODO

    def set_single_inpt(self, inpt):
        inpt_height = self.from_shape[0] * self.tiles[0]
        inpt_width = self.from_shape[1] * self.tiles[1]
        output_height = self.to_shape[0] * self.tiles[0]
        output_width = self.to_shape[1] * self.tiles[1]

        self.single_inpt = inpt.reshape((inpt_height, inpt_width))
        shaped_output = T.zeros((output_height, output_width))
        for i in xrange(self.tiles[0]):
            for j in xrange(self.tiles[1]):
                inpt_tile = self.single_inpt[
                            i * self.from_shape[1]:(i + 1) * self.from_shape[1],
                            j * self.from_shape[0]:(j + 1) * self.from_shape[0]] \
                    .flatten()

                output_tile_flat = self.activation_fn(T.dot(inpt_tile, self.ws[i * self.tiles[1] + j]))

                # output_tile = T.reshape(output_tile_flat, self.to_shape)
                output_tile = T.reshape(output_tile_flat, (self.to_shape[0], self.to_shape[1]))

                shaped_output = T.set_subtensor(shaped_output[
                                                i * self.to_shape[0]:(i + 1) * self.to_shape[0],
                                                j * self.to_shape[1]:(j + 1) * self.to_shape[1]],
                                                output_tile)

        self.single_output = T.reshape(shaped_output, (self.n_out,)) + self.b

    #         self.single_inpt = inpt.reshape((self.n_in, ))
    #         self.single_output = self.activation_fn(
    #             (1 - self.p_dropout) * T.dot(self.single_inpt, self.w) + self.b)


    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return -T.mean((self.output - y) ** 2)

    def cost(self, net):
        "Return the cost."
        return T.mean((self.output_dropout - net.y) ** 2)


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)
