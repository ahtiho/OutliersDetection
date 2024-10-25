import os
import numpy as np
import pandas as pd
from data_fetch import combine_data
from AutoEncoders import VAE, train, evaluate, plot_reconstruction
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = 'vae_model.pth'

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

def main():
    try:
        df = pd.read_csv('files/transactions.csv')
        raw_data = df.copy()
        if df.shape[0] > 0:
            print("Data loaded successfully!")
        else:
            raise ValueError("No data to load!")
        encoded_data = encode_labels(raw_data)
        columns_to_exclude = ['Unnamed: 0', 'transaction_id', 'authorized_date', 'date']
        data = encoded_data.drop(columns=columns_to_exclude)
        data = data.dropna()
        model = VAE(input_dim=12, hidden_dim=36, latent_dim=2)
        data = torch.tensor(data.values, dtype=torch.float32)
        if torch.isnan(data).any() or torch.isinf(data).any() or not np.issubdtype(raw_data.values.dtype, np.number):
            raise ValueError("Input data contains NaN values!")
        data = TensorDataset(data)
        data_loader = DataLoader(data, batch_size=32, shuffle=True)
        
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH))
            print("Model loaded from disk.")
        else:
            train(model, data_loader, epochs=100)
            torch.save(model.state_dict(), MODEL_PATH)
            print("Model trained and saved to disk.")
        
        original, reconstructed, errors = evaluate(model, data_loader, df)
        plot_reconstruction(original, reconstructed, errors)
        
    except FileNotFoundError:
        print("No data to load or file not found!")
        data = combine_data()
        print(data.head())
        data.to_csv('transactions.csv', index=False)
        print("Data fetched and saved to transactions.csv")
        print(data.head())

if __name__ == "__main__":
    main()