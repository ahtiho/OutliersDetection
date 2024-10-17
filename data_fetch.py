import pandas as pd
import numpy as np
import torch.nn as nn
import plaid
from plaid.api import plaid_api
from plaid.model.transactions_get_request import TransactionsGetRequest
from dotenv import load_dotenv
import os
from datetime import date
from synthetic_data import generate_synthetic_transactions

def load_environment_variables():
    load_dotenv()

def configure_plaid_client():
    configuration = plaid.Configuration(
        host=plaid.Environment.Sandbox,
        api_key={'clientId': os.getenv('PLAID_CLIENT_ID'), 'secret': os.getenv('PLAID_SECRET')}
    )
    client = plaid_api.PlaidApi(plaid.ApiClient(configuration))
    return client

def get_transactions(client, start_date, end_date):
    request = TransactionsGetRequest(access_token=os.getenv('access_token'), start_date=start_date, end_date=end_date)
    response = client.transactions_get(request)
    transactions = response['transactions']
    return transactions

def flatten_transactions(transactions):
    flattened_data_list = []
    for transaction_data in transactions:
        flattened_data = {
            'account_id': transaction_data['account_id'],
            'amount': transaction_data['amount'],
            'authorized_date': transaction_data['authorized_date'],
            #'category': ', '.join(transaction_data['category']),  # Flatten the category list
            'category_id': transaction_data['category_id'],
            'date': transaction_data['date'],
            'iso_currency_code': transaction_data['iso_currency_code'],
            'merchant_name': transaction_data['merchant_name'],
            'name': transaction_data['name'],
            'payment_channel': transaction_data['payment_channel'],
            'pending': transaction_data['pending'],
            'personal_finance_primary_category': transaction_data['personal_finance_category']['primary'],
            'personal_finance_detailed_category': transaction_data['personal_finance_category']['detailed'],
            'personal_finance_confidence_level': transaction_data['personal_finance_category']['confidence_level'],
            'transaction_id': transaction_data['transaction_id'],
            'transaction_type': transaction_data['transaction_type']
        }
        flattened_data_list.append(flattened_data)
    df = pd.DataFrame(flattened_data_list)
    return df

def combine_data():
    load_environment_variables()
    client = configure_plaid_client()
    start_date = date(2021, 1, 1)
    end_date = date(2024, 12, 31)
    
    try:
        transactions = get_transactions(client, start_date, end_date)
        df = flatten_transactions(transactions)
        df_2 = generate_synthetic_transactions(1000)
        
        df = pd.concat([df, df_2], ignore_index=True)
        
        df.to_csv('transactions.csv')
        return df
        
    except Exception as e:
        print(f"An error occurred while fetching transactions: {e}")
        print("Try again...")

