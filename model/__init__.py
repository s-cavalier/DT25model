import numpy as np

print('--------------------------------')
print('GPU related speedup/errors/info.')
print('--------------------------------')

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
class TimeSeriesModel:
    def create_sequences(data, n_steps_in, n_steps_out):
        """
        Create sequences of past time steps (n_steps_in) to predict future time steps (n_steps_out)
        """
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(data):
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def __init__(self):
        self.model = tf.keras.models.load_model('model/saved_model.h5')
        self.model.summary()
    
    def predict(self, data: np.ndarray,mode):
        """Takes in 10 np arrays of size 22, outputs bool"""

        # print("Passes here")
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled_X = scaler.fit_transform(data)
        # tensor_X = scaled_X.reshape(1, 10, 21)
        # prediction = self.model.predict(tensor_X, batch_size = 1)
        # return prediction[0][0]
        if (mode == True):
            batch, timesteps, features = data.shape
        
            # Reshape to 2D for scaling (shape: (batch*timesteps, features))
            data_2d = data.reshape(batch * timesteps, features)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_2d = scaler.fit_transform(data_2d)
            
            # Reshape back to (1, 10, 21)
            scaled_data = scaled_2d.reshape(batch, timesteps, features)
            
            # Run model prediction
            prediction = self.model.predict(scaled_data, batch_size=1)
            return prediction[0][0]
        
        else:
            data = data.pd.Dataframe(data)
            data.dropna()
            data["Timestamp"] = data["Timestamp"].str.strip('"')
            data["Timestamp"] = pd.to_datetime(data["Timestamp"])
            data = data.to_numpy(dtype= np.float32)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_X = scaler.fit_transform(data)
            tensor_X = scaled_X.reshape(1, 10, 20)
            prediction = self.model.predict(tensor_X, batch_size = 1)
            return prediction[0][0]
        
        


print('--------------')
print('Booting model.')
print('--------------')
SINGLETON = TimeSeriesModel()
