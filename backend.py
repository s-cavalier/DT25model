import socket
import json
import requests
import numpy as np
from collections import deque
from sklearn.preprocessing import MinMaxScaler

# TCP Client Configuration
TCP_HOST = "localhost"
TCP_PORT = 30004
MAX_ENTRIES = 10  # Store last 10 entries
robot_data_buffer = deque(maxlen=MAX_ENTRIES)

# FastAPI Model Endpoint
FASTAPI_URL = "http://localhost:9054/model"

def preprocess_data_for_model():
    """Prepares the last 10 RTDE entries into the correct format for the model"""
    if len(robot_data_buffer) < 10:
        print("\n⚠️ Not enough data yet. Waiting for 10 entries...")
        return None

    # Convert deque to a list and then to a NumPy array
    df = json.loads(json.dumps(list(robot_data_buffer)))

    # Extract the necessary features (should be 21 total)
    feature_columns = [
        "Timestamp","Current_J0", "Current_J1", "Current_J2", "Current_J3", "Current_J4", "Current_J5",
        "Temperature_T0", "Temperature_T1", "Temperature_T2", "Temperature_T3", "Temperature_T4", "Temperature_T5",
        "Speed_J0", "Speed_J1", "Speed_J2", "Speed_J3", "Speed_J4", "Speed_J5",
        "Tool_current", "robot_fail"  # 21 features
    ]

    # Ensure feature count is correct
    if len(feature_columns) != 21:
        print("\n❌ Error: Feature column count mismatch! Expected 21, but got", len(feature_columns))

    df_filtered = np.array([[entry[col] for col in feature_columns] for entry in df])

    # Debug print for feature shape
    print("\n🔹 Shape before scaling:", df_filtered.shape)  # Should be (10, 21)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_X = scaler.fit_transform(df_filtered)

    # Debug print for scaled data shape
    print("\n🔹 Shape after scaling:", scaled_X.shape)  # Should still be (10, 21)

    # Reshape to (1, 10, 21) for LSTM model input
    formatted_data = scaled_X.reshape(1, 10, 21)

    # Debug print for final shape
    print("\n✅ Final shape before sending to model:", formatted_data.shape)  # Should be (1, 10, 21)
    print(formatted_data)
    return formatted_data.tolist()  # Convert to list for JSON serialization


def send_to_model():
    """Send formatted RTDE data to the FastAPI model"""
    data = preprocess_data_for_model()
    if data is None:
        return  # Not enough data yet

    payload = {
        "VALIDATOR": "10-SEQUENCE",
        "DATAFRAME": data  # Properly formatted input
    }

    try:
        response = requests.post(FASTAPI_URL, json=payload)
        if response.status_code == 200:
            print("\n🔹 Model Response:", response.json())
        else:
            print("\n❌ Model Error:", response.status_code, response.text)
    except requests.exceptions.RequestException as e:
        print("\n⚠️ Request to model failed:", e)

def connect_to_tcp_server():
    """Connect to the TCP mock server and receive RTDE data"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((TCP_HOST, TCP_PORT))
        print(f"🔹 Connected to TCP Server at {TCP_HOST}:{TCP_PORT}")

        try:
            while True:
                data = client_socket.recv(1024).decode("utf-8")  # Receive data from TCP server
                if not data:
                    break
                
                try:
                    json_data = json.loads(data.strip())  # Convert JSON string to Python dictionary
                    robot_data_buffer.append(json_data)  # Store last 10 entries

                    #print("\n🔹 Stored Robot Data (Last 10 Entries):")
                    #for entry in list(robot_data_buffer):
                    #    print(entry)

                    # Send to FastAPI model once we have 10 entries
                    if len(robot_data_buffer) == 10:
                        send_to_model()

                except json.JSONDecodeError:
                    print("❌ Received malformed JSON:", data)
        except KeyboardInterrupt:
            print("\n🔹 Client shutting down.")
        finally:
            client_socket.close()

if __name__ == "__main__":
    connect_to_tcp_server()
