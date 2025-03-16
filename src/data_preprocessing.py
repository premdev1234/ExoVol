
# Handles data loading, preprocessing, feature engineering, and normalization.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['time_to_expiry'] = (df['expiration'] - df['date']).dt.days
    
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    encoder = LabelEncoder()
    df['call_put'] = encoder.fit_transform(df['call_put'])
    df['act_symbol'] = encoder.fit_transform(df['act_symbol'])
    
    features = ['strike', 'bid', 'ask', 'vol', 'delta', 'gamma', 'theta', 'vega', 'rho', 'time_to_expiry']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, scaler
