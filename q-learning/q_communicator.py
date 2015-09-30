"""
Image Protocol:
---------

A stream of RGB-images is received on a UDP socket.
The image is flattened to a 1D-array of 3*(`image_length` ** 2) bytes and 
partitioned into `n_fragments` fragments, each of length `fragment_length`.
Each fragment gets an additional byte at position 0, specifying the fragment 
number of the corresponding fragment. This allows the image to be constructed
without major inaccuracies even if packets get lost or the order is changed.

"""

import socket
import threading
import SocketServer
import time
import struct

import matplotlib.pyplot as plt
import numpy as np
from q_learner import QLearner, QNetwork
from labeling_network import FullyConnectedLayer, linear, Network


import livebarchart
from safetyrule import SafetyRule


# Period in which Q-Function is adjusted
learning_period = 0.01
time_next_learning = time.time() + learning_period

# Period in which decisions are made and state is stored to exp-store
decision_period = 0.075
time_next_decision = time.time() + decision_period

random_action_duration = 0.75
random_action_termination = time.time()

q_network_save_filename = 'saved-q-networks/q-network'
q_network_save_period = 5.0 # time in seconds
q_network_next_save_time = time.time() + q_network_save_period

#q_network_load_filename = None
q_network_load_filename = 'saved-q-networks/q-network_trained'

MB_SIZE = 10
EXP_STORE_SIZE = 50000
PERCEPT_LENGTH = 25
N_ACTIONS = 3
STATE_STM = 4
GAMMA = 0.98
LEARNING_RATE = 0.0008
LEARNING_ITERATIONS_PER_UPDATE = 80
BURN_IN = 50

EPSILON_START = 1.0
EPSILON_END = 0.00
EPSILON_DECREASE_DURATION = 1600.0

Q_HIDDEN_NEURONS = 200

prng = np.random.RandomState(1234567)


safety_rules = [SafetyRule(vector_start=0,
                           vector_stop=5,
                           threshold=0.15,
                           safety_action=3)]
current_safety_action = None


labeling_network_file_name = 'saved-nns/best_encoder_bigdata'
labeling_net = Network.load_from_file(labeling_network_file_name, MB_SIZE)


if q_network_load_filename:
    q_function = QNetwork.load_from_file(q_network_load_filename, MB_SIZE)
else:
    hidden_layer = FullyConnectedLayer(STATE_STM*PERCEPT_LENGTH,
                                Q_HIDDEN_NEURONS)
    output_layer = FullyConnectedLayer(Q_HIDDEN_NEURONS,
                                N_ACTIONS,
                                activation_fn=linear)
    q_function = QNetwork([hidden_layer, output_layer], minibatch_size=MB_SIZE)

q_learner = QLearner(q_function,
                     exp_store_size=EXP_STORE_SIZE,
                     percept_length=PERCEPT_LENGTH,
                     n_actions=N_ACTIONS,
                     state_stm=STATE_STM,
                     gamma=GAMMA,
                     minibatch_size=MB_SIZE,
                     prng=prng)

bar_plotter = livebarchart.LiveBarPlotter(n_categories=5,
                                         n_bars_per_category=5)

PORT = 8888
IP = "0.0.0.0"

REMOTE_HOST = "127.0.0.1"
REMOTE_PORT = 8889

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

n_fragments = 2
fragment_length = 32*64*3
image_length = 64
data = [np.zeros(fragment_length, dtype=np.uint8)] * n_fragments
data_reward = 0.

class UDPHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        current_data = self.request[0]
        if len(current_data) != fragment_length + 5:
            print 'WRONG DATA LENGTH!', len(current_data)
            return
        global data
        global data_reward
        array = np.fromstring(current_data, dtype=np.uint8)
        data_reward = struct.unpack('f', array[1:5])[0]
        data[array[0]] = array[5:]


class ThreadedUDPServer(SocketServer.ThreadingMixIn, SocketServer.UDPServer):
    pass


def draw_image(figure, figure_image, img_array):
    figure_image.set_data(img_array)
    figure.canvas.draw()


def send_data(x):
    sock.sendto(chr(int(x) % 256), (REMOTE_HOST, REMOTE_PORT))


def learn(q_learner, learning_rate):
    q_learner.train_q_function(learning_rate)


def remember_and_decide(percept, last_action, previous_reward, epsilon, prng,
                        n_actions, safety_rules):
    
    encoding = labeling_net.get_single_output(percept)
    q_learner.add_observation(encoding.astype(np.float32),
                              last_action,
                              previous_reward)
    bar_plotter.update(encoding)

    for safety_rule in safety_rules:
        global current_safety_action;
        current_safety_action = safety_rule.check_percept(encoding)

    try:
        print 'qs:', q_learner.get_current_qs()
    except:
        print 'not enough states'

    global random_action_termination
    if time.time() < random_action_termination:
        return last_action

    if prng.uniform(0,1) > epsilon \
            and q_learner.exp_store_current_size > BURN_IN:
        action = q_learner.get_current_best_action()
    else:
        action = prng.randint(n_actions)
        random_action_termination = time.time() + random_action_duration

    return action


if __name__ == "__main__":
    ReceiverSocket = ThreadedUDPServer((IP,PORT), UDPHandler)
    ServerThread = threading.Thread(target=ReceiverSocket.serve_forever)
    ServerThread.daemon = True
    ServerThread.start()


    # Set up image plot.
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    img_d = ax.imshow(np.random.randint(0, 256, (64, 64, 3)),
#                      interpolation='nearest')
#    fig.canvas.draw()
#    plt.show(block=False)

    total_frames = 0
    timer_start = time.time()
    training_start = time.time()

    current_img_array = np.asarray(
        np.random.randint(0, 256, n_fragments*fragment_length),
        np.uint8)
        
    last_action = 0
    previous_reward_total = 0

    while 1:
        plt.pause(0.0001)
        # Assemble image. TODO: performance (not every frame)
        for i in range(n_fragments):
            current_img_array[i*fragment_length : (i+1)*fragment_length] = data[i]
        
        
        if time.time() > time_next_learning \
                and q_learner.exp_store_current_size > BURN_IN:
            time_next_learning = time.time() + learning_period
            for it in xrange(LEARNING_ITERATIONS_PER_UPDATE):
                learn(q_learner, LEARNING_RATE)
    
    
#        if time.time() > time_next_random_action:
#            time_next_random_action = time.time() + random_action_change_period
#            current_random_action = prng.randint(N_ACTIONS)


        if time.time() > time_next_decision:
            time_next_decision = time.time() + decision_period
            
            previous_reward = data_reward - previous_reward_total
            previous_reward_total = data_reward
            
            epsilon = max(EPSILON_END,
                          EPSILON_START - \
                            (EPSILON_START - EPSILON_END) * \
                            (time.time()-training_start)/EPSILON_DECREASE_DURATION)
            
            
            last_action = remember_and_decide(current_img_array,
                                              last_action,
                                              previous_reward,
                                              epsilon,
                                              prng,
                                              N_ACTIONS,
                                              safety_rules)

            if current_safety_action == None:
                send_data(last_action)
            else:
                send_data(current_safety_action)
            
            print 'epsilon: {0:.3}, action: {1}'.format(epsilon, last_action)
    
#        draw_image(fig, img_d, 
#            current_img_array.reshape((image_length, image_length, 3)))
#        plt.pause(0.01)

        if time.time() > q_network_next_save_time:
            q_network_next_save_time = time.time() + q_network_save_period
            q_learner.q_function.save_as_file(q_network_save_filename)

        time.sleep(0.01)
        
        if total_frames % 10 == 0:
#            print data_reward
            pass

        # FPS      
        total_frames += 1
        if total_frames % 100 == 0:
            print np.sum(current_img_array)
            print 'Current fps: %f' % (100.0/(time.time() - timer_start))
            timer_start = time.time()
