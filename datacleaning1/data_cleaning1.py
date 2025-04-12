import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os  # Added os import

class DataCleaner:
    def __init__(self):
        """Initialize the data cleaning process"""
        print("Initializing Data Cleaning Process...")
        self.create_directories()
        self.load_data()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = ['cleaned_data', 'raw_data']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

    def load_data(self):
        """Load all datasets"""
        try:
            # Load datasets
            self.financial_metrics = pd.read_csv('../financial_metrics.csv')
            self.transactions = pd.read_csv('../transactions.csv')
            self.users = pd.read_csv('../users.csv')
            
            print("\nInitial Data Overview:")
            print(f"Financial Metrics: {len(self.financial_metrics)} rows")
            print(f"Transactions: {len(self.transactions)} rows")
            print(f"Users: {len(self.users)} rows")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def clean_financial_metrics(self):
        """Clean financial metrics dataset"""
        print("\nCleaning Financial Metrics...")
        df = self.financial_metrics.copy()
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Add derived metrics
        df['net_profit'] = df['revenue'] - df['expenses']
        df['profit_margin_calculated'] = (df['net_profit'] / df['revenue'] * 100).round(2)
        df['working_capital_ratio'] = (df['current_ratio'] * df['working_capital']).round(2)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['metric_id'])
        
        self.cleaned_financial_metrics = df
        return df

    def clean_transactions(self):
        """Clean transactions dataset"""
        print("\nCleaning Transactions...")
        df = self.transactions.copy()
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate total tax and related features
        df['total_tax'] = df['cgst'] + df['sgst'] + df['igst'] + df['tds']
        df['total_tax_calculated'] = df.apply(lambda x: x['amount'] * 0.18, axis=1)  # Expected tax at 18% GST
        df['tax_discrepancy'] = df['total_tax'] - df['total_tax_calculated']
        
        # Add amount categories based on transaction size
        conditions = [
            (df['amount'] < 1000),
            (df['amount'] >= 1000) & (df['amount'] < 5000),
            (df['amount'] >= 5000) & (df['amount'] < 10000),
            (df['amount'] >= 10000)
        ]
        choices = ['Small', 'Medium', 'Large', 'Very Large']
        df['transaction_amount_category'] = np.select(conditions, choices, default='Unknown')
        
        # Add time-based features
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Clean categorical columns
        categorical_cols = ['transaction_type', 'account_type', 'payment_method', 'category', 'status']
        for col in categorical_cols:
            df[col] = df[col].str.strip().str.title()
        
        # Handle missing values
        df['notes'] = df['notes'].fillna('')
        df['attachments'] = df['attachments'].fillna(0)
        df['recurring_frequency'] = df['recurring_frequency'].fillna('None')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['transaction_id'])
        
        self.cleaned_transactions = df
        return df

    def clean_users(self):
        """Clean users dataset"""
        print("\nCleaning Users...")
        df = self.users.copy()
        
        # Convert dates
        date_cols = ['joining_date', 'last_login']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
        
        # Calculate derived fields
        df['experience_years'] = ((datetime.now() - df['joining_date'])
                                .dt.total_seconds() / (365.25 * 24 * 60 * 60)).round(2)
        
        # Clean text fields
        text_cols = ['department', 'role', 'status']
        for col in text_cols:
            df[col] = df[col].str.strip().str.title()
        
        # Handle missing values
        df['access_level'] = df['access_level'].fillna(1)
        df['approval_limit'] = df['approval_limit'].fillna(df.groupby('role')['approval_limit'].transform('median'))
        
        # Add risk scoring
        role_risk = {
            'Executive': 2,
            'Manager': 3,
            'Admin': 4,
            'Developer': 2,
            'Analyst': 1
        }
        df['role_risk_score'] = df['role'].map(role_risk).fillna(1)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id'])
        
        self.cleaned_users = df
        return df

    def save_cleaned_data(self):
        """Save cleaned datasets"""
        self.cleaned_financial_metrics.to_csv('cleaned_data/cleaned_financial_metrics.csv', index=False)
        self.cleaned_transactions.to_csv('cleaned_data/cleaned_transactions.csv', index=False)
        self.cleaned_users.to_csv('cleaned_data/cleaned_users.csv', index=False)
        print("\nCleaned data saved in 'cleaned_data' directory")

    def generate_cleaning_report(self):
        """Generate a cleaning report"""
        report = "=== Data Cleaning Report ===\n\n"
        
        datasets = {
            'Financial Metrics': (self.financial_metrics, self.cleaned_financial_metrics),
            'Transactions': (self.transactions, self.cleaned_transactions),
            'Users': (self.users, self.cleaned_users)
        }
        
        for name, (original, cleaned) in datasets.items():
            report += f"{name}:\n"
            report += f"Original rows: {len(original)}\n"
            report += f"Cleaned rows: {len(cleaned)}\n"
            report += f"Rows removed: {len(original) - len(cleaned)}\n"
            report += f"Missing values filled: {original.isnull().sum().sum()}\n\n"
        
        with open('reports/cleaning_report.txt', 'w') as f:
            f.write(report)
        
        print("\nCleaning report saved in 'reports/cleaning_report.txt'")
    
    def clean_all_data(self):
        """Execute complete cleaning process"""
        print("\nStarting data cleaning process...")
        
        # Clean all datasets
        self.clean_financial_metrics()
        self.clean_transactions()
        self.clean_users()
        
        # Save cleaned data and report
        self.save_cleaned_data()
        self.generate_cleaning_report()
        
        print("\nData cleaning process completed successfully!")

def main():
    """Main execution function"""
    try:
        cleaner = DataCleaner()
        cleaner.clean_all_data()
    except Exception as e:
        print(f"\nError in data cleaning process: {str(e)}")
        raise

if __name__ == "__main__":
    main()