import socket

TCP_IP = "0.0.0.0"  
TCP_PORT = 12345
BUFFER_SIZE = 1024

tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_server.bind((TCP_IP, TCP_PORT))
tcp_server.listen(1)

print("Waiting for connection...")
conn, addr = tcp_server.accept()
print(f"Connected to {addr}")

try:
    while True:
        data = conn.recv(BUFFER_SIZE)
        if not data:
            break
        print("Received coordinates:")
        print(data.decode().strip())
except KeyboardInterrupt:
    print("Stopping server...")
finally:
    conn.close()
    tcp_server.close()