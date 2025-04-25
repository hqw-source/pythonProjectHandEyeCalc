import socket
import threading
import time


class Server:
    global_client_sock = None
    dataRec = ''
    recInit = 0
    Flag = False
    client_sock_flag = 0

    # 处理客户端连接
    def handle_client(self, client_socket):
        self.global_client_sock = client_socket
        while True:
            try:
                self.dataRec = client_socket.recv(1024)
                if not self.dataRec:
                    print("客户端已主动断开连接")
                    break

                self.dataRec = self.dataRec.decode('utf-8')
                print("收到数据:", self.dataRec)
                self.recInit += 1
                self.Flag = True

            except (ConnectionResetError, ConnectionAbortedError):
                print("连接被重置或中止")
                break
            except OSError as e:
                print("套接字关闭或出错:", e)
                break

            if not self.client_sock_flag:
                print("收到主线程请求断开客户端")
                break

        client_socket.close()
        print("客户端处理线程退出")

    # 接受客户端连接
    def accept_connections(self):
        while self.accepting:
            try:
                client_sock, addr = self.server_socket.accept()
                self.global_client_sock = client_sock
                self.client_sock_flag = True
                print(f"客户端 {addr} 已连接")

                client_handler = threading.Thread(target=self.handle_client, args=(client_sock,), daemon=True)
                client_handler.start()
            except OSError as e:
                print("accept_connections 已终止，socket 被关闭:", e)
                break  # socket 被关闭，跳出循环

    # 发送数据到客户端
    def send_data(self, data):
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

        # 关闭客户端 socket
        if self.global_client_sock:
            try:
                self.global_client_sock.shutdown(socket.SHUT_RDWR)
            except Exception as e:
                print("套接字关闭或出错:", e)
            try:
                self.global_client_sock.close()
                print("客户端连接已关闭")
            except Exception as e:
                print("关闭客户端连接失败:", e)
            self.global_client_sock = None
        else:
            print("没有可关闭的客户端连接")

        # 关闭服务器 socket
        if hasattr(self, "server_socket") and self.server_socket:
            self.accepting = False  # <-- 停止 accept 循环
            try:
                self.server_socket.close()
                print("服务器socket已关闭")
            except Exception as e:
                print("关闭服务器socket失败:", e)
            self.server_socket = None

        # 新增：关闭服务器socket，防止端口占用
        if hasattr(self, "server_socket") and self.server_socket:
            try:
                self.server_socket.close()
                print("服务器socket已关闭")
            except Exception as e:
                print("关闭服务器socket失败:", e)
            self.server_socket = None

    def start_server(self, ipArr, ipPort):
        if hasattr(self, 'server_socket') and self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_sock_flag = False

        try:
            self.server_socket.bind((ipArr, ipPort))
        except OSError as e:
            print(f"绑定端口失败：{e}")
            return

        self.server_socket.listen(5)
        print(f"服务器启动，监听端口 {ipPort}...")

        self.accepting = True  # <-- 添加控制标志
        self.thread = threading.Thread(target=self.accept_connections, daemon=True)
        self.thread.start()


if __name__ == '__main__':
    # 启动服务器
    server = Server()
    server.start_server('127.0.0.1', 8000)

    # 保持服务器运行
    try:
        while True:
            time.sleep(10)

            server.closeclient()
    except KeyboardInterrupt:
        print("服务器被手动停止")
        server.closeclient()
        print("服务器端连接已关闭")
