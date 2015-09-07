'''
    Simple udp socket server
    Silver Moon (m00n.silv3r@gmail.com)
'''
 
import socket
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import struct
 
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
 

fig = plt.figure()
ax = fig.add_subplot(111)
img_d = ax.imshow(np.random.uniform(0.0, 1.0, (64, 64, 3)), 
                  interpolation='nearest')

fig.canvas.draw()
plt.show(block=False)


fragment_pointer = -1
n_fragments = 4*3
receive_floats = False
current_graph = [[]] * n_fragments



def draw_image_from_string(img):
    img_d.set_data(
        np.asarray(
            [np.fromstring(i, dtype=np.uint8) for i in img]).reshape((64, 64, 3)))
    fig.canvas.draw()

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


while 1:
    d = s.recvfrom(1024)
    data = d[0]
    addr = d[1]
    print 'Message[' + addr[0] + ':' + str(addr[1]) + '] - length ' + str(len(data))
     
    if is_preamble(data):
        print 'Received preamble.'
        if fragment_pointer != (-1):
            print 'Incomplete packet!'
        fragment_pointer = 0
    else:
        print 'Received data.'
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

