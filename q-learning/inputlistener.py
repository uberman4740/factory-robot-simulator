#!/usr/bin/env python

"""inputlistener.py: Threaded UPD-listener, writing into global storage."""
import threading
import SocketServer

IP = '127.0.0.1'
PORT = 8888
current_data = []


def collect_current_data():
    global current_data
    d = []
    while len(current_data) > 0:
        d.append(current_data.pop())
    return d


class UDPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        global current_data
        current_data.append(self.request[0])


class ThreadedUDPServer(SocketServer.ThreadingMixIn, SocketServer.UDPServer):
    pass


ReceiverSocket = ThreadedUDPServer((IP, PORT), UDPHandler)
ServerThread = threading.Thread(target=ReceiverSocket.serve_forever)
ServerThread.daemon = True
ServerThread.start()
