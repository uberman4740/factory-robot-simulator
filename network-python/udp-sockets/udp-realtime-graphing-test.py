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

# some X and Y data
n_bars = 5
x = np.arange(n_bars)
y = np.random.uniform(0, 1, n_bars)

ax.set_ylim(0,1)
ax.set_xlim(-0.5, n_bars-0.5)
rects = ax.bar(x, y, align='center')

periods = np.random.uniform(1, 3, n_bars)
ranges = np.random.uniform(0.2,1.0, n_bars)

# draw and show it
fig.canvas.draw()
plt.show(block=False)


#now keep talking with the client
while 1:
    # receive data from client (data, addr)
    d = s.recvfrom(1024)
    data = d[0]
    addr = d[1]
     
    if not data: 
        break

    try:
        # floats = struct.unpack('f', data[0:4])
        floats = struct.unpack('%sf' % (len(data)/4), data)
        print 'floats:', floats

        # y += np.random.uniform(-0.05, 0.05, 5)
        y = np.zeros(5)
        y[:len(floats)] = floats

        # set the new data
        for rect, h in zip(rects, y):
            rect.set_color('#1177aa')
            rect.set_height(h)
            

        fig.canvas.draw()

        time.sleep(0.01)
    except KeyboardInterrupt:
        break

    # reply = 'OK...' + data
    reply = data 

    s.sendto(reply , addr)
    print 'Message[' + addr[0] + ':' + str(addr[1]) + '] - ' + data.strip()
     
s.close()