import numpy as np

print('--------------------------------')
print('GPU related speedup/errors/info.')
print('--------------------------------')

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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
    
    def predict(self, data: np.ndarray):
        """Takes in 10 np arrays of size 22, outputs bool"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_X = scaler.fit_transform(data)
        tensor_X = scaled_X.reshape(1, 10, 21)
        prediction = self.model.predict(tensor_X, batch_size = 1)
        return prediction[0][0]


print('--------------')
print('Booting model.')
print('--------------')
SINGLETON = TimeSeriesModel()
