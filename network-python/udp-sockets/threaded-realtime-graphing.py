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
import matplotlib.pyplot as plt

import numpy as np

PORT = 8888
IP = "0.0.0.0"

n_fragments = 2
fragment_length = 32*64*3
image_length = 64
data = [np.zeros(fragment_length, dtype=np.uint8)] * n_fragments


class UDPHandler(SocketServer.BaseRequestHandler):

    def handle(self):
        current_data = self.request[0]
        if len(current_data) != fragment_length + 1:
            print 'WRONG DATA LENGTH!', len(current_data)
            return
        global data
        array = np.fromstring(current_data, dtype=np.uint8)
        data[array[0]] = array[1:]


class ThreadedUDPServer(SocketServer.ThreadingMixIn, SocketServer.UDPServer):
    pass


def draw_image(figure, figure_image, img_array):
    figure_image.set_data(img_array)
    figure.canvas.draw()


if __name__ == "__main__":
    ReceiverSocket = ThreadedUDPServer((IP,PORT), UDPHandler)
    ServerThread = threading.Thread(target=ReceiverSocket.serve_forever)
    ServerThread.daemon = True
    ServerThread.start()

    correct_count = 0
    violation_count = 0

    # Set up image plot.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img_d = ax.imshow(np.random.randint(0, 256, (64, 64, 3)), 
                      interpolation='nearest')
    fig.canvas.draw()
    plt.show(block=False)

    total_frames = 0
    timer_start = time.time()

    current_img_array = np.asarray(
        np.random.randint(0, 256, n_fragments*fragment_length),
        np.uint8)

    while 1:
        for i in range(n_fragments):
            current_img_array[i*fragment_length : (i+1)*fragment_length] = data[i]
        draw_image(fig, img_d, 
            current_img_array.reshape((image_length, image_length, 3)))

        # FPS      
        total_frames += 1
        if total_frames % 100 == 0:
            print 'Current fps: %f' % (100.0/(time.time() - timer_start))
            timer_start = time.time()
