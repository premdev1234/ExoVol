"""
VAE-LSTM model combining latent space encoding and time-series forecasting.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed

def build_vae_lstm(vae_encoder, lstm_model, time_steps, features):
    """
    Combines a pretrained VAE encoder with an LSTM for time-series forecasting.
    """
    latent_dim = vae_encoder.output_shape[1]
    
    # LSTM input layer
    lstm_inputs = Input(shape=(time_steps, latent_dim))
    x = LSTM(128, return_sequences=True, activation='tanh')(lstm_inputs)
    x = LSTM(64, activation='tanh')(x)
    output = Dense(features, activation='linear')(x)
    
    model = Model(inputs=lstm_inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model
