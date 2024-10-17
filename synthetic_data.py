from faker import Faker
import random
import pandas as pd
from datetime import datetime, timedelta


fake = Faker()

def generate_synthetic_transactions(n):
    transactions = []
    for _ in range(n):
        transaction = {
            'account_id': fake.uuid4(),  # Unique account ID
            'amount': round(random.uniform(-500, 1000), 2),  # Random amount between 1 and 1000
            'authorized_date': (datetime.now() - timedelta(days=random.randint(0, 365))).date(),  # Random authorized date within the last year
            'category_id': str(random.randint(10000000, 13005000)),  # Random category ID
            'date': (datetime.now() - timedelta(days=random.randint(0, 365))).date(),  # Random transaction date within the last year
            'iso_currency_code': random.choice(['USD', 'EUR', 'GBP']),  # Random currency code
            'merchant_name': fake.company(),  # Random merchant name
            'name': fake.word(),  # Random transaction name
            'payment_channel': random.choice(['online', 'in store', 'other']),  # Random payment channel
            'pending': random.choice([True, False]),  # Random pending status
            'personal_finance_primary_category': random.choice(['GENERAL_MERCHANDISE', 'FOOD_AND_DRINK', 'HEALTH_CARE']),
            'personal_finance_detailed_category': random.choice(['GENERAL_MERCHANDISE_OTHER', 'RESTAURANTS', 'MEDICAL_SERVICES']),
            'personal_finance_confidence_level': random.choice(['LOW', 'MEDIUM', 'HIGH']),  # Confidence level
            'transaction_id': fake.uuid4(),  # Unique transaction ID
            'transaction_type': random.choice(['place', 'digital', 'transfer'])  # Random transaction type
        }
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)