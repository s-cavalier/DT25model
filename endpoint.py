import numpy as np
import pandas as pd
from model import SINGLETON
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://datathon25.so-cavalier.com",
        "https://datathon25.so-cavalier.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"]
)

@app.post("/model")
async def model(body: dict):
    """Receives RTDE data from backend.py and predicts using the model."""
    if "VALIDATOR" not in body:
        return Response(status_code=404, content="Bad request.")
    if body["TYPE"] == "True":
        df1 = pd.DataFrame(body["DATAFRAME"])
        df1.dropna()
        df1["Timestamp"] = df1["Timestamp"].str.strip('"')
        df1["Timestamp"] = pd.to_datetime(df1["Timestamp"])
        
        body = { 
            "VALIDATOR": "10-SEQUENCE", 
            "DATAFRAME": [df1.to_numpy(dtype = np.float32).tolist()]
        }
        #print(body["DATAFRAME"])
    if body["VALIDATOR"] == "10-SEQUENCE":
        print("\n🔹 Received Data from Backend")
        
        # Directly convert the incoming data to a NumPy array.
        # This preserves the (1, 10, 21) shape expected by your LSTM.
        LSTM_input = np.array(body["DATAFRAME"], dtype=np.float32)
        
        print("✅ Data Preprocessing Complete")
        print("\n🔹 Received Data Shape from Backend:", LSTM_input.shape)

        # Check that the shape matches the expected (1, 10, 21)
        if LSTM_input.shape != (1, 10, 20):
            print("\n❌ Shape Mismatch: Expected (1, 10, 21), Got", LSTM_input.shape)
            return {
                "VALIDATOR": "ERROR",
                "TYPE": "Shape mismatch.",
                "DETAILS": str(LSTM_input.shape)
            }

        # Pass input to the model
        print("hello")
        #need to change mode based on what is needed, True is for continuous, false is for csv
        output = SINGLETON.predict(LSTM_input, True)  # Already shaped correctly
        print("passes at the output")
        prediction_result = {
            "VALIDATOR": "GOOD",
            "RAW-OUTPUT": float(output),
            "ROUND-OUTPUT": int(output + 0.5)
        }

        # Print Model Predictions
        print("\n🔹 Model Prediction:")
        print(f"➡ Raw Output: {prediction_result['RAW-OUTPUT']}")
        print(f"➡ Rounded Output: {prediction_result['ROUND-OUTPUT']}")
        print("-" * 50)

        return prediction_result

    return {"VALIDATOR": "ERROR", "TYPE": "Unknown validator."}

if __name__ == "__main__":
    print("---------------")
    print("Model loaded.")
    print("Booting server.")
    print("---------------")
    
    uvicorn.run(app, host="0.0.0.0", port=9054)
