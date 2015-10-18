import os
import ntpath
import matplotlib.pyplot as plt
import numpy as np


outputfilename = 'plt.png'
rootdir = 'trained-networks/'
valid_filename = 'decoder_learning_curve_bigdatavalidation_accuracies.lcurve'
train_filename = 'decoder_learning_curve_bigdatatrain_costs.lcurve'

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def produce_plot(path, ax):
    data = read_data(path)
    ax.plot(data, label=path)


def read_data(path):
    f = open(path)
    lines = f.readlines()
    f.close()

    return np.asarray([float(l) for l in lines])


if __name__ == '__main__':

    fig = plt.figure(figsize=(30, 20))

    ax = fig.add_subplot(111)

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == train_filename or file == valid_filename:
                # print os.path.join(subdir, file)
                produce_plot(os.path.join(subdir, file), ax)


    ax.legend()
    fig.savefig(outputfilename)