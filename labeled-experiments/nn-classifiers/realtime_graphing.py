'''
Based on:

    Simple udp socket server
    Silver Moon (m00n.silv3r@gmail.com)
'''
 
import socket
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import struct

import labeling_network as lbln
from labeling_network import Network


network_file_names = ['stored-networks/monochrome_humanoid_classifier__00523/humanoid_net_best_larger',
                      'stored-networks/pickupbox_classifier__01/pickup_box_classifier_best']

trained_nets = [Network.load_from_file(file_name, 10) for file_name in network_file_names]
 
HOST = ''   # Symbolic name meaning all available interfaces
PORT = 8888 # Arbitrary non-privileged port
 
# Datagram (udp) socket
try :
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print 'Socket created'
except socket.error, msg :
    print 'Failed to create socket. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
    sys.exit()
 
 
# Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error , msg:
    print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
    sys.exit()
     
print 'Socket bind complete'
 

# fig = plt.figure()
# ax = fig.add_subplot(111)
# img_d = ax.imshow(np.random.uniform(0.0, 1.0, (64, 64, 3)), 
#                   interpolation='nearest')

# fig.canvas.draw()


chart_fig = plt.figure()
chart_ax = chart_fig.add_subplot(111)
chart_ax.set_xlim(-2.5, 2.5)
chart_ax.set_ylim(0.0, 1.0)

bar_width = 0.2
n_bars = len(trained_nets)

charts = []
chart_colors = ['#882222', '#3399bb', '#55aa22']
for i in range(n_bars):
    charts.append(chart_ax.bar(np.arange(5)-2 + (i - n_bars/2.0 + 0.5)*bar_width, np.ones(5), bar_width, color=chart_colors[i],  align='center'))
    # chart2 = chart_ax.bar(np.arange(5)-2+0.5*bar_width, np.ones(5)/2, bar_width, color='#882222',  align='center')

chart_fig.canvas.draw()





plt.show(block=False)


fragment_pointer = -1
n_fragments = 4*3
receive_floats = False
current_graph = [[]] * n_fragments



def draw_image_from_string(img):
    arr = np.asarray(
            [np.fromstring(i, dtype=np.uint8) for i in img]).reshape((64, 64, 3))
    # img_d.set_data(arr)
    # fig.canvas.draw()

    float_pixels = arr.reshape(12288) / 255.0

    predictions = [net.get_single_output(float_pixels) for net in trained_nets]
    # print predictions

    for chart, prediction in zip(charts, predictions):
        for rect, p in zip(chart, prediction):
            rect.set_height(p)

    chart_fig.canvas.draw()


def draw_image_from_floats(img):
    img_d.set_data(np.asarray(img).reshape((64, 64, 3)))
    fig.canvas.draw()



def is_preamble(data):
    if len(data) < 6: 
        return False
    for i in range(6):
        if ord(data[i]) != (i+1):
            return False
    return True

incomplete_packet_count = 0
while 1:
    d = s.recvfrom(1024)
    data = d[0]
    addr = d[1]
    # print 'Message[' + addr[0] + ':' + str(addr[1]) + '] - length ' + str(len(data))
     
    if is_preamble(data):
        # print 'Received preamble.'
        if fragment_pointer != (-1):
            incomplete_packet_count += 1
            print 'Incomplete packet! count: %d' % incomplete_packet_count
        fragment_pointer = 0
    else:
        # print 'Received data.'
        if fragment_pointer >= 0:
            if receive_floats:
                floats = struct.unpack('%sf' % (len(data)/4), data)
                current_graph[fragment_pointer] = floats
            else:
                current_graph[fragment_pointer] = data

            fragment_pointer += 1
            if fragment_pointer >= n_fragments:
                if receive_floats:
                    draw_image_from_floats(current_graph)
                else:
                    draw_image_from_string(current_graph)
                fragment_pointer = -1

    # reply = 'OK...' + data
    # s.sendto(reply , addr)
    
s.close()

