"""
Sending python object over network based on socket and pickle.
Firstly has to be run server,
then client will connect automaticly.
"""

import socket
import select
import threading
import pickle
import time

HEADERSIZE = 10


class Server:
    """
    Starts a socket server on given port, optionaly on given ip - in other case server listen on every adresses.
    Last receivered object from any client is stored in LAST_MESSAGE.
    For send object to every connected client --> call method send_message(<the_object>)
    """
    def __init__(self, port, ip="", welcome=None):
        """
        Create server
        Takes port, optional ip (else listening on each), welcome object - send after connection
            - socket_list contains every connected client and server
            - clients contains every connected client
            - statusLevel - 0-no console status, 1-types, 2-whole message
            - LAST_MESSAGE stores last receivered object
            - receivering is blocked by select if there is no incomming message - runs in separated thread
        """
        self.S = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.S.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.S.bind((ip, port))

        self.S.listen()

        self.sockets_list = [self.S]
        self.clients = []

        self.counter_sent = 0
        self.counter_recievered = 0
        self.statusLevel = 0
        self.WELCOME = welcome
        self.LAST_MESSAGE = None

        self.t = threading.Thread(target=self.main)
        self.t.start()

        print(f'Listening for connections on port {port}...')

    @staticmethod
    def receive_message(client_sock):
        """
        Takes one argument - object of client which is sending the message.
        Reconstructing message: incomming message has to contain header with int, which tells how long message is,
                                rest of the message is decoded to object by pickle.
        """
        full_msg = b''
        new_msg = True
        while True:
            try:
                msg = client_sock.recv(16)
            except Exception as e:
                print("Receiver error:", e)
                return None
            if new_msg:
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

            full_msg += msg

            if len(full_msg)-HEADERSIZE == msglen:
                full_msg = full_msg[HEADERSIZE:]
                full_msg = pickle.loads(full_msg)
                return full_msg

    def send_message(self, message):
        """
        Sending object to every connected cliend.
        Message is attached to header which containes long of message.
        """
        for cli_sock in self.clients:
            if message is not None:
                pickedmessage = pickle.dumps(message)
                message_header = f"{len(pickedmessage):<{HEADERSIZE}}".encode('utf-8')
                cli_sock.send(message_header + pickedmessage)

                self.counter_sent += 1

                if self.statusLevel == 1:
                    print(f"message send, type: {type(message).__name__}")
                elif self.statusLevel == 2:
                    print(f"message receivered:\n{message}")
                return True
            else:
                return False

    def main(self):
        """
        Runs in separated thread - manage client list and receivering objects.
        """
        while True:
            read_sockets, _, exception_sockets = select.select(self.sockets_list, [], self.sockets_list)

            for notified_socket in read_sockets:
                if notified_socket == self.S:
                    client_socket, client_address = self.S.accept()

                    self.sockets_list.append(client_socket)
                    self.clients.append(client_socket)

                    print('Accepted new connection from {}:{}'.format(*client_address))

                    if self.WELCOME is not None:
                        self.send_message(self.WELCOME)
                        if self.statusLevel == 1:
                            print(f"welcome sent, type: {type(self.WELCOME).__name__}")
                        elif self.statusLevel == 2:
                            print(f"welcome sent:\n{self.WELCOME}")

                else:
                    last_message = self.receive_message(notified_socket)
                    if last_message is not None:
                        self.LAST_MESSAGE = last_message
                    else:
                        print('Closed connection from: {}:{}'.format(*client_address))

                        self.sockets_list.remove(notified_socket)
                        self.clients.remove(notified_socket)
                        continue

                    self.counter_recievered += 1
                    if self.statusLevel == 1:
                        print(f"message receivered, type: {type(self.LAST_MESSAGE).__name__}")
                    elif self.statusLevel == 2:
                        print(f"message receivered:\n{self.LAST_MESSAGE}")

            for notified_socket in exception_sockets:
                self.sockets_list.remove(notified_socket)


class Client:
    """
    Automaticly connecting to a socket server on given ip, port.
    Last receivered object from server is stored in LAST_MESSAGE.
    For send object to server --> call method send_message(<the_object>)
    """
    def __init__(self, ip, port):
        """
        - LAST_MESSAGE stores last receivered object
        - receivering is blocked by recv method if there is no incomming message - runs in separated thread
        """
        self.IP = ip
        self.PORT = port
        self.retry = 5

        self.connect()

        self.counter_sent = 0
        self.counter_recievered = 0
        self.statusLevel = 0
        self.LAST_MESSAGE = None

        self.t = threading.Thread(target=self.main)
        self.t.start()

    def connect(self):
        """
        Connection to server, handel exceptions - reconnecting
        """
        self.S = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.S.connect((self.IP, self.PORT))
                print(f"Connection to {self.IP}:{self.PORT} succesfull!")
                break
            except ConnectionRefusedError as e:
                print(str(e.__class__.__name__) + ": " + str(e))
                print(f"connection unsuccesfull, reconnect in {self.retry}s")
                time.sleep(self.retry)

    def recv_message(self):
        """
        Handel exceptions - call connect for reconnecting
        Reconstructing message: incomming message has to contain header with int, which tells how long message is,
                                rest of the message is decoded to object by pickle.
        """
        full_msg = b''
        new_msg = True
        while True:
            try:
                msg = self.S.recv(1024)
            except ConnectionResetError as e:
                print(e)
                print(f"in {self.retry}s reconect...")
                time.sleep(self.retry)
                self.S.close()
                self.connect()
                continue
            except ConnectionAbortedError as e:
                return False

            if new_msg:
                msglen = int(msg[:HEADERSIZE])
                new_msg = False

            full_msg += msg

            if len(full_msg)-HEADERSIZE == msglen:
                full_msg = full_msg[HEADERSIZE:]
                full_msg = pickle.loads(full_msg)
                return full_msg
            elif len(full_msg)-HEADERSIZE > msglen:

                print("přeplněný buffer")
                return None

    def send_message(self, message):
        """
        Sending object to every server.
        Message is attached to header which containes long of message.
        """
        if message:
            messagepickled = pickle.dumps(message)
            message_header = f"{len(messagepickled):<{HEADERSIZE}}".encode('utf-8')
            self.S.send(message_header + messagepickled)

            self.counter_sent += 1

            if self.statusLevel == 1:
                print(f"message send, type: {type(message).__name__}")
            elif self.statusLevel == 2:
                print(f"message receivered:\n{message}")

            return True
        else:
            return False

    def main(self):
        """
        Runs in separated thread - manage receivering objects.
        """
        while True:
            self.LAST_MESSAGE = self.recv_message()

            if type(self.LAST_MESSAGE).__name__ == "bool":
                if not self.LAST_MESSAGE:
                    break

            self.counter_recievered += 1

            if self.statusLevel == 1:
                print(f"message receivered, type: {type(self.LAST_MESSAGE).__name__}")
            elif self.statusLevel == 2:
                print(f"message receivered:\n{self.LAST_MESSAGE}")
