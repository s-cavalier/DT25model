import socket
import json
import time
import random

# TCP Server Configuration
TCP_HOST = "0.0.0.0"  # Listen on all interfaces
TCP_PORT = 30004  # Port for TCP communication

def get_mock_robot_data():
    """Generate mock RTDE data"""
    return {
        "Timestamp": int(time.time() * 1000),

        "Current_J0": round(random.uniform(0, 10), 2),
        "Current_J1": round(random.uniform(0, 10), 2),
        "Current_J2": round(random.uniform(0, 10), 2),
        "Current_J3": round(random.uniform(0, 10), 2),
        "Current_J4": round(random.uniform(0, 10), 2),
        "Current_J5": round(random.uniform(0, 10), 2),

        "Temperature_T0": round(random.uniform(20, 80), 2),
        "Temperature_T1": round(random.uniform(20, 80), 2),
        "Temperature_T2": round(random.uniform(20, 80), 2),
        "Temperature_T3": round(random.uniform(20, 80), 2),
        "Temperature_T4": round(random.uniform(20, 80), 2),
        "Temperature_T5": round(random.uniform(20, 80), 2),

        "Speed_J0": round(random.uniform(-180, 180), 2),
        "Speed_J1": round(random.uniform(-180, 180), 2),
        "Speed_J2": round(random.uniform(-180, 180), 2),
        "Speed_J3": round(random.uniform(-180, 180), 2),
        "Speed_J4": round(random.uniform(-180, 180), 2),
        "Speed_J5": round(random.uniform(-180, 180), 2),

        "Tool_current": round(random.uniform(0, 5), 2),
        "robot_fail": random.choice([0, 1])
    }

def start_tcp_server():
    """Start a TCP server that sends mock RTDE data"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((TCP_HOST, TCP_PORT))
        server_socket.listen(1)
        print(f"🔹 TCP Server started on {TCP_HOST}:{TCP_PORT}")

        conn, addr = server_socket.accept()
        print(f"🔹 Client connected: {addr}")

        try:
            while True:
                robot_data = get_mock_robot_data()
                json_data = json.dumps(robot_data) + "\n"  # Newline to separate messages
                conn.sendall(json_data.encode("utf-8"))  # Send JSON as bytes
                time.sleep(1)  # Send data every second
        except BrokenPipeError:
            print("❌ Client disconnected.")
        except KeyboardInterrupt:
            print("\n🔹 Shutting down TCP server.")
        finally:
            conn.close()

if __name__ == "__main__":
    start_tcp_server()
