from faker import Faker
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

# Constants
ACCOUNT_TYPES = ['SAVINGS', 'DAILY_USE', 'INVESTMENTS']
CURRENCIES = ['USD', 'EUR', 'GBP']
PAYMENT_CHANNELS = ['online', 'in store', 'other']
PRIMARY_CATEGORIES = ['GENERAL_MERCHANDISE', 'FOOD_AND_DRINK', 'HEALTH_CARE']
DETAILED_CATEGORIES = ['GENERAL_MERCHANDISE_OTHER', 'RESTAURANTS', 'MEDICAL_SERVICES']
CONFIDENCE_LEVELS = ['LOW', 'MEDIUM', 'HIGH']
TRANSACTION_TYPES = ['place', 'digital', 'transfer']

class SyntheticTransactionGenerator:
    def __init__(self):
        self.fake = Faker()

    def _generate_transaction(self) -> Dict[str, any]:
        return {
            'account_id': random.choice(ACCOUNT_TYPES),
            'amount': round(random.uniform(-500, 1000), 2),
            'authorized_date': (datetime.now() - timedelta(days=random.randint(0, 365))).date(),
            'category_id': str(random.randint(10000000, 13005000)),
            'date': (datetime.now() - timedelta(days=random.randint(0, 365))).date(),
            'iso_currency_code': random.choice(CURRENCIES),
            'merchant_name': self.fake.company(),
            'name': self.fake.word(),
            'payment_channel': random.choice(PAYMENT_CHANNELS),
            'pending': random.choice([True, False]),
            'personal_finance_primary_category': random.choice(PRIMARY_CATEGORIES),
            'personal_finance_detailed_category': random.choice(DETAILED_CATEGORIES),
            'personal_finance_confidence_level': random.choice(CONFIDENCE_LEVELS),
            'transaction_id': self.fake.uuid4(),
            'transaction_type': random.choice(TRANSACTION_TYPES)
        }

    def generate_synthetic_transactions(self, n: int) -> pd.DataFrame:
        transactions: List[Dict[str, any]] = [self._generate_transaction() for _ in range(n)]
        return pd.DataFrame(transactions)

def main():
    generator = SyntheticTransactionGenerator()
    df = generator.generate_synthetic_transactions(100)
    print(df.head())

if __name__ == "__main__":
    main()