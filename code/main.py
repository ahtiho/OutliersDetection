import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from data_fetch import TransactionProcessor, PlaidClient
from AutoEncoders import VAE, train, evaluate, plot_reconstruction, plot_anomalies, plot_latent_space

MODEL_PATH = 'vae_model_2.pth'
DATA_PATH = 'files/transactions.csv'
EXCLUDED_COLUMNS = ['transaction_id', 'authorized_date', 'date']

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
    
    return data_tensor

def load_or_train_model(train_loader):
    model = VAE(input_dim=12, hidden_dim=36, latent_dim=2)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Model loaded from disk.")
    else:
        train(model, train_loader, epochs=100)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model trained and saved to disk.")
    return model

def main():
    df = load_data(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=np.random.randint(0, 10000))

    train_tensor = preprocess_data(train_df)
    test_tensor = preprocess_data(test_df)
        
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=32, shuffle=False)
    
    model = load_or_train_model(train_loader)
    original, reconstructed, errors, outlier_df = evaluate(model, test_loader, test_df)
    test_df['reconstruction_error'] = errors
    plot_anomalies(test_df)
    plot_reconstruction(errors)
    
    outlier_df = outlier_df.drop(columns = ['outlier', 'reconstruction_error'] )
    outlier_tensor = preprocess_data(outlier_df)
    print(outlier_df.head())
    print(test_df.head())
    plot_latent_space(model, outlier_tensor)
    

if __name__ == "__main__":
    main()