import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from data_fetch import TransactionProcessor, PlaidClient
from AutoEncoders import VAE, train, evaluate, plot_reconstruction

MODEL_PATH = 'vae_model.pth'
DATA_PATH = 'files/transactions.csv'
EXCLUDED_COLUMNS = ['Unnamed: 0', 'transaction_id', 'authorized_date', 'date']

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.shape[0] > 0:
            print("Data loaded successfully!")
            return df
        else:
            raise ValueError("No data to load!")
    except FileNotFoundError:
        print("No data to load or file not found!")
        client = PlaidClient()
        data = TransactionProcessor.combine_data(client)
        data.to_csv(file_path, index=False)
        print("Data fetched and saved to transactions.csv")
        return data

def encode_labels(raw_data):
    non_numeric_cols = raw_data.select_dtypes(include=['object', 'bool']).columns
    label_encoders = {}
    for column in non_numeric_cols:
        le = LabelEncoder()
        raw_data[column] = le.fit_transform(raw_data[column].astype(str))
        label_encoders[column] = le
    
    if not np.issubdtype(raw_data.values.dtype, np.number):
        raise ValueError("Data contains non-numeric values after encoding")
    
    return raw_data

def preprocess_data(df):
    raw_data = df.copy()
    encoded_data = encode_labels(raw_data)
    data = encoded_data.drop(columns=EXCLUDED_COLUMNS)
    data = data.dropna()
    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    
    if torch.isnan(data_tensor).any() or torch.isinf(data_tensor).any():
        raise ValueError("Input data contains NaN or Inf values!")
    
    return TensorDataset(data_tensor)

def load_or_train_model(data_loader):
    model = VAE(input_dim=12, hidden_dim=36, latent_dim=2)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Model loaded from disk.")
    else:
        train(model, data_loader, epochs=100)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model trained and saved to disk.")
    return model

def main():
    df = load_data(DATA_PATH)
    data_loader = DataLoader(preprocess_data(df), batch_size=32, shuffle=True)
    model = load_or_train_model(data_loader)
    original, reconstructed, errors = evaluate(model, data_loader, df)
    plot_reconstruction(original, reconstructed, errors)

if __name__ == "__main__":
    main()