#!/usr/bin/env python

"""sensordecoder.py: Conversion of raw data to convenient format, according to protocol."""

import struct
import numpy as np

from log import log


class SensorDecoder(object):
    def __init__(self,
                 n_fragments,
                 n_checksum_bytes,
                 frame_counter_position,
                 fragment_id_position,
                 img_data_position,
                 img_fragment_length,
                 action_position,
                 reward_position,
                 n_reward_bytes=4):
        self.img_data = [None] * n_fragments
        self.reward_data = 0.0
        self.action_data = 0
        self.n_fragments = n_fragments
        self.receipt_table = [False] * n_fragments
        self.n_checksum_bytes = n_checksum_bytes
        self.frame_counter_position = frame_counter_position
        self.fragment_id_position = fragment_id_position
        self.img_data_position = img_data_position
        self.img_fragment_length = img_fragment_length
        self.action_position = action_position
        self.reward_position = reward_position
        self.n_reward_bytes = n_reward_bytes
        self.current_frame_counter = 0

    def add_data(self, data, current_frame_counter):
        """Add fragment to the decoder's buffer."""
        if current_frame_counter != self.current_frame_counter:
            self.current_frame_counter = current_frame_counter
            self.receipt_table = [False] * self.n_fragments
        array = np.fromstring(data, dtype=np.uint8)
        if check_integrity(array, self.n_checksum_bytes):
            fc = array[self.frame_counter_position]
            if fc != current_frame_counter:
                log('Wrong frame counter. (Expected {0} got {1})'.format(current_frame_counter, fc))
                return
            fragment_id = array[self.fragment_id_position]
            log('Frag: {0}, FC: {1}'.format(fragment_id, fc))

            self.receipt_table[fragment_id] = True
            self.reward_data = struct.unpack('f', array[self.reward_position:
                                                        self.reward_position + self.n_reward_bytes])[0]
            self.action_data = array[self.action_position]
            self.img_data[fragment_id] = array[self.img_data_position:-self.n_checksum_bytes]
        else:
            log('Checksum error!')

    def get_current_data(self):
        """Returns triple of current flattened image, action and reward."""
        img_array = np.empty(self.n_fragments * self.img_fragment_length,
                             dtype=np.uint8)
        for i in xrange(self.n_fragments):
            img_array[i * self.img_fragment_length: (i + 1) * self.img_fragment_length] = self.img_data[i]
        return img_array, self.action_data, self.reward_data

    def state_info_complete(self):
        """Test whether all fragments of the frame are present."""
        return self.receipt_table == [True] * self.n_fragments


def check_integrity(data, n_checksum_bytes):
    """Checks whether checksum matches data."""
    if len(data) < n_checksum_bytes:
        return False
    return np.mod(np.sum(data[:-n_checksum_bytes]), 256) == reduce(lambda x, y: 256 * x + y,
                                                                   np.nditer(data[-n_checksum_bytes:]))
