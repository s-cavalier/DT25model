import numpy as np
import pandas as pd
from model import SINGLETON
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://datathon25.so-cavalier.com","https://datathon25.so-cavalier.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"]
)

@app.post("/model")
async def model(body: dict):
    if "VALIDATOR" not in body:
        return Response(status_code=404, content='Bad request.')
    if body["VALIDATOR"] == "10-SEQUENCE":
        df = pd.DataFrame(body["DATAFRAME"])
        df.dropna()
        df['Timestamp'] = df['Timestamp'].str.strip('"')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        LSTM_input = df.iloc[1:, :].to_numpy(dtype=np.float32)
        output = SINGLETON.predict(LSTM_input)
        return { "VALIDATOR" : "GOOD", "RAW-OUTPUT" : float(output), "ROUND-OUTPUT" : int(output + 0.5) }
    return { "VALIDATOR" : "ERROR", "TYPE" : "Unknown validator." }

if __name__ == '__main__':
    print('---------------')
    print('Model loaded.')
    print('Booting server.')
    print('---------------')
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9054)