import pandas as pd
import os

print("Checking transaction file date ranges...\n")

# Primary file check
primary_path = 'anomaly_results/complete_transactions_with_anomalies.csv'
if os.path.exists(primary_path):
    print(f"Reading {primary_path}...")
    df1 = pd.read_csv(primary_path)
    date_col = pd.to_datetime(df1['date'])
    print(f"Primary file date range: {date_col.min().strftime('%Y-%m-%d')} to {date_col.max().strftime('%Y-%m-%d')}")
    print(f"Number of transactions: {len(df1)}")
    
    # Look for any older transactions (2019)
    old_dates = df1[pd.to_datetime(df1['date']).dt.year < 2020]
    if len(old_dates) > 0:
        print(f"\nFound {len(old_dates)} transactions from before 2020:")
        for i, row in old_dates.head(5).iterrows():
            print(f"- {row['transaction_id']}: {row['date']}")
    else:
        print("\nNo transactions found from before 2020.")
else:
    print(f"File not found: {primary_path}")

# Secondary file check
secondary_path = 'datacleaning1/anomaly_results/complete_transactions_with_anomalies.csv'
if os.path.exists(secondary_path):
    print(f"\nReading {secondary_path}...")
    try:
        df2 = pd.read_csv(secondary_path)
        date_col = pd.to_datetime(df2['date'])
        print(f"Secondary file date range: {date_col.min().strftime('%Y-%m-%d')} to {date_col.max().strftime('%Y-%m-%d')}")
        print(f"Number of transactions: {len(df2)}")
        
        # Look for any older transactions (2019)
        old_dates = df2[pd.to_datetime(df2['date']).dt.year < 2020]
        if len(old_dates) > 0:
            print(f"\nFound {len(old_dates)} transactions from before 2020:")
            for i, row in old_dates.head(5).iterrows():
                print(f"- {row['transaction_id']}: {row['date']}")
        else:
            print("\nNo transactions found from before 2020 in secondary file.")
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"\nFile not found: {secondary_path}")

# Also check fraud report for dates
fraud_path = 'fraud_results/fraud_report.csv'
if os.path.exists(fraud_path):
    print(f"\nReading fraud report: {fraud_path}...")
    try:
        df3 = pd.read_csv(fraud_path)
        date_col = pd.to_datetime(df3['date'])
        print(f"Fraud report date range: {date_col.min().strftime('%Y-%m-%d')} to {date_col.max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error reading fraud report: {e}")
else:
    print(f"\nFraud report not found: {fraud_path}")

print("\nDate range check completed.") 