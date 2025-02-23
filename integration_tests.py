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
    print(SINGLETON.predict(LSTMinput))

def ping_server(dataframe: pd.DataFrame):
    data = dataframe.to_dict(orient='list')  # Converts to list of dicts (JSON-like format)
    body = { "VALIDATOR" : "10-SEQUENCE", "DATAFRAME" : data }

    # Send POST request with JSON data
    response = requests.post("http://localhost:9054/model", json=body)

    # Print response
    print(response.status_code)
    print(response.json())  # Assuming the API returns JSON

df = pd.read_excel("dataset.xlsx") # Toy example
df.dropna()
df_subset = df.iloc[0:11, 1:22]  

ping_server(df_subset)