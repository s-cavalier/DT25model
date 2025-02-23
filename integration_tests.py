import pandas as pd
import numpy as np
from random import randint
import requests

"""
Very approximate tests
"""

def static_access(dataframe: pd.DataFrame):
    from model import SINGLETON

    rows = randint(2, 7000)
    df_subset = dataframe.iloc[rows:(rows + 10), 1:22]  
    LSTMinput = df_subset.to_numpy(dtype=np.float32)

    # Now pass it to the model
    print(SINGLETON.predict(LSTMinput,False))

# def ping_server(dataframe: pd.DataFrame):
#     data = dataframe.to_dict(orient='list')  # Converts to list of dicts (JSON-like format)
#     body = { "VALIDATOR" : "10-SEQUENCE", "DATAFRAME" : data }

#     # Send POST request with JSON data
#     response = requests.post("http://localhost:9054/model", json=body)

#     # Print response
#     print(response.status_code)
#     print(response.json())  # Assuming the API returns JSON
def ping_server(dataframe: pd.DataFrame):
    # Ensure that the DataFrame has exactly 10 rows and 21 columns.
    # Adjust slicing if needed. Here, we take the first 10 rows and columns 1 to 22.
    df_subset = dataframe.iloc[:10, 0:22]
    df_subset['Timestamp'] = df_subset['Timestamp'].str.strip('"')

    df_subset['Timestamp'] = pd.to_datetime(df_subset['Timestamp'])
    # Convert the DataFrame to a NumPy array (should be of shape (10,21))
    arr = df_subset.to_numpy(dtype=np.float32)
    
    # Wrap the 2D array in an outer list to create a 3D array (shape: (1,10,21))
    body = { 
        "VALIDATOR": "10-SEQUENCE", 
        "DATAFRAME": [arr.tolist()] 
    }
    
    # Send POST request with JSON data
    response = requests.post("http://localhost:9054/model", json=body)
    
    # Print response
    print("Status code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception as e:
        print("Failed to decode JSON:", e)

df = pd.read_excel("dataset.xlsx") # Toy example
df.dropna()
df_subset = df.iloc[0:11, 1:22]  

ping_server(df_subset)