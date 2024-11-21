import socket
import threading

class Server:
    global_client_sock = None
    dataRec = ''
    recInit = 0
    Flag = False
    client_sock_flag=0
    # 处理客户端连接
    def handle_client(self, client_socket):
        self.global_client_sock = client_socket
        while True:
            try:
                self.dataRec = client_socket.recv(1024)
                if not self.dataRec:
                    break
                self.dataRec = self.dataRec.decode('utf-8')
                print("收到数据:", self.dataRec)
                self.recInit += 1
                self.Flag = True
            except (ConnectionResetError, OSError):
                break
            if self.client_sock_flag is False:
                break
        client_socket.close()

    # 接受客户端连接
    def accept_connections(self):
        # while True:
            client_sock, addr = self.server_socket.accept()
            self.client_sock_flag = True
            print("收到来自 %s 的连接" % str(addr))
            client_handler = threading.Thread(target=self.handle_client, args=(client_sock,))
            client_handler.start()
            # break

    # 发送数据到客户端
    def send_data(self, data):
        self.dataRec = ''
        if self.global_client_sock:
            try:
                self.global_client_sock.sendall(data.encode('utf-8'))
                print("发送数据:", data)
            except BrokenPipeError:
                print("发送失败，客户端连接已关闭")
        else:
            print("没有客户端连接，无法发送数据")

    # 关闭当前客户端连接而不退出服务器
    def closeclient(self):
        self.client_sock_flag = False
        if self.global_client_sock:
            self.global_client_sock.close()
            print("客户端连接已关闭")
            self.global_client_sock = None  # 清空当前客户端引用，准备接受新连接
        else:
            print("没有可关闭的客户端连接")

    def start_server(self, ipArr, ipPort):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sock_flag = False
        # 绑定端口
        self.server_socket.bind((ipArr, ipPort))
        # 设置最大连接数，超过后排队
        self.server_socket.listen(5)
        print(f"服务器启动，监听端口 {ipPort}...")
        # 在新线程中接受连接
        self.thread = threading.Thread(target=self.accept_connections)
        self.thread.start()
