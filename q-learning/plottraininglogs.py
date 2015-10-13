import os
import matplotlib.pyplot as plt
import numpy as np

rootdir = 'training-logs/'
output_name = 'rewards.png'


# REWARD_SMOOTHING = 0.999615
REWARD_SMOOTHING = 0.9998
FPS = 30.0
Y_MIN = -1.0
Y_MAX = 1.0


def produce_log_plot(path, smoothing_factor, multiplier):
    data = read_data(path)
    # print 'data', data
    smooth_data = smooth_accumulate_data(data, smoothing_factor, multiplier)
    # print
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Average reward w.r.t. time.')
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel('reward')
    ax.plot(smooth_data)
    fig.savefig(os.path.join(os.path.dirname(path), output_name))


def read_data(path):
    f = open(path)
    lines = f.readlines()
    f.close()

    return np.asarray([float(l) for l in lines])


def smooth_accumulate_data(data, smoothing_factor, multiplier=1.0):
    smooth_data = []
    acc = 0.0
    for d in data:
        acc = smoothing_factor*acc + (1.-smoothing_factor)*d*multiplier
        smooth_data.append(acc)
    return smooth_data

if __name__ == '__main__':
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == 'rewards.log':
                # print os.path.join(subdir, file)
                produce_log_plot(os.path.join(subdir, file), REWARD_SMOOTHING, FPS)
