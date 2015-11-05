import numpy as np
import theano

data_path_labels = '../../../dev-repos/factory-robot-data/imgs_2015-10-25/labels.dat'
data_path_actions = '../../../dev-repos/factory-robot-data/imgs_2015-10-25/actions.dat'
#
# data_path_labels = '../../../dev-repos/factory-robot-data/imgs_2015-10-18/labels.dat'
# data_path_actions = '../../../dev-repos/factory-robot-data/imgs_2015-10-18/actions.dat'


def canonical_vec(i, n):
    result = [0] * n
    result[i] = 1
    return result


def canonical_vec_multiple(l, n):
    result = []
    for a in l:
        long_form = [0] * n
        long_form[a] = 1
        result.extend(long_form)
    return result


def get_data(lower, upper, n_actions=3, prediction_interval=1, data_path_lab=None, data_path_act=None):
    if (data_path_lab is None) or (data_path_actions is None):
        import warnings
        warnings.warn('Using get_data without specifying the data paths is discouraged.')

    if data_path_lab is not None:
        data_path_labels_ = data_path_lab
    else:
        data_path_labels_ = data_path_labels

    if data_path_act is not None:
        data_path_actions_ = data_path_act
    else:
        data_path_actions_ = data_path_actions

    f_labels = open(data_path_labels_)
    data_lines_labels = f_labels.readlines()
    assert len(data_lines_labels) > upper - lower
    data_lines_labels = data_lines_labels[lower:upper]
    f_labels.close()

    f_actions = open(data_path_actions_)
    data_lines_actions = f_actions.readlines()
    assert len(data_lines_actions) > upper - lower
    data_lines_actions = np.asarray(data_lines_actions[lower+1: upper+1], dtype='int16')
    f_actions.close()

    data_lines_actions = np.repeat(data_lines_actions[:, np.newaxis],
                                   prediction_interval,
                                   axis=1)
    for t in xrange(1, prediction_interval):
        data_lines_actions[:-t, t:t+1] = data_lines_actions[t:, t:t+1]
        data_lines_actions[-t:, t] = np.zeros(t)

    data = [[x for x in l_l.split(',')[:-1]] + canonical_vec_multiple(a, n_actions)
            for l_l, a in zip(data_lines_labels, data_lines_actions)]
    return np.asarray(data, dtype=theano.config.floatX)


def make_batches(data, sequence_length, prediction_interval=1, n_actions=3, crop_end=0, input_images=None):
    crop_end += prediction_interval
    result_input = []
    result_percept = []
    result_input_images = []
    for i in xrange((data.shape[0]-prediction_interval) / sequence_length):
        result_input.append(data[i*sequence_length: (i+1)*sequence_length - crop_end])
        result_percept.append(data[i*sequence_length+prediction_interval:
                                   (i+1)*sequence_length+prediction_interval - crop_end,
                                   :-(n_actions*prediction_interval)])
        if input_images is not None:
            result_input_images.append(input_images[i*sequence_length: (i+1)*sequence_length - crop_end])

    if input_images is not None:
        return np.asarray(result_input), np.asarray(result_percept), np.asarray(result_input_images)
    return np.asarray(result_input), np.asarray(result_percept)

if __name__ == '__main__':
    print get_data(4, 50)
