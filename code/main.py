
import numpy as np
import pandas as pd
from data_fetch import combine_data

def main():
    try:
        data = pd.read_csv('transactions.csv')
        if data.shape[0] > 0:
            print("Data loaded successfully!")
        else:
            raise ValueError("No data to load!")
    except (FileNotFoundError, ValueError):
        print("No data to load or file not found!")
        data = combine_data()
        data.to_csv('transactions.csv', index=False)
        print("Data fetched and saved to transactions.csv")
        print(data.head())

if __name__ == "__main__":
    main()