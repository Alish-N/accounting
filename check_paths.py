import os
import glob
import pandas as pd

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Check paths in anomaly_results/
print("\nChecking paths relative to current directory:")
path1 = "anomaly_results/complete_transactions_with_anomalies.csv"
print(f"- Path: {path1}")
print(f"- File exists: {os.path.exists(path1)}")

# Check paths in datacleaning1/anomaly_results/
print("\nChecking alternate path:")
path2 = "datacleaning1/anomaly_results/complete_transactions_with_anomalies.csv"
print(f"- Path: {path2}")
print(f"- File exists: {os.path.exists(path2)}")

# List all CSV files in both directories
print("\nListing all CSV files in anomaly_results/:")
print(glob.glob("anomaly_results/*.csv"))

print("\nListing all CSV files in datacleaning1/anomaly_results/:")
print(glob.glob("datacleaning1/anomaly_results/*.csv"))

# Check if directories exist
print("\nChecking if directories exist:")
print(f"- anomaly_results directory exists: {os.path.isdir('anomaly_results')}")
print(f"- datacleaning1/anomaly_results directory exists: {os.path.isdir('datacleaning1/anomaly_results')}")

# Check file dates in both locations
print("\n=== File Content Analysis ===")
if os.path.exists(path1):
    try:
        df1 = pd.read_csv(path1)
        print(f"\nFile at {path1}:")
        print(f"- Number of rows: {len(df1)}")
        print(f"- Date range: {min(df1['date'])} to {max(df1['date'])}")
        print(f"- Sample transaction IDs: {df1['transaction_id'].head(3).tolist()}")
    except Exception as e:
        print(f"Error reading {path1}: {str(e)}")

if os.path.exists(path2):
    try:
        df2 = pd.read_csv(path2)
        print(f"\nFile at {path2}:")
        print(f"- Number of rows: {len(df2)}")
        print(f"- Date range: {min(df2['date'])} to {max(df2['date'])}")
        print(f"- Sample transaction IDs: {df2['transaction_id'].head(3).tolist()}")
    except Exception as e:
        print(f"Error reading {path2}: {str(e)}")

# Check if the files are duplicates
if os.path.exists(path1) and os.path.exists(path2):
    try:
        size1 = os.path.getsize(path1)
        size2 = os.path.getsize(path2)
        print(f"\nAre the files identical?")
        print(f"- File sizes: {path1}={size1} bytes, {path2}={size2} bytes")
        print(f"- Size match: {size1 == size2}")
        
        if 'df1' in locals() and 'df2' in locals():
            print(f"- Row count match: {len(df1) == len(df2)}")
            if len(df1) == len(df2):
                sample_match = (df1['transaction_id'].head(5) == df2['transaction_id'].head(5)).all()
                print(f"- Sample transaction ID match: {sample_match}")
    except Exception as e:
        print(f"Error comparing files: {str(e)}")

# Path used in FraudDetector class
print("\n=== FraudDetector Configuration ===")
print("Path in FraudDetector.__init__: self.transactions_file = 'anomaly_results/complete_transactions_with_anomalies.csv'")
print("Path in app.py (detect_fraud): anomaly_results_path = 'anomaly_results/complete_transactions_with_anomalies.csv'")

# Recommendation
print("\n=== Recommendation ===")
if os.path.exists(path1):
    print("Use the path: 'anomaly_results/complete_transactions_with_anomalies.csv'")
    print("This matches the path in the code and contains valid data.")
elif os.path.exists(path2):
    print("Consider updating the path in the code to use: 'datacleaning1/anomaly_results/complete_transactions_with_anomalies.csv'")
    print("The file exists at this location but not at the path specified in the code.")
else:
    print("No valid transaction file found. You need to create the file at: 'anomaly_results/complete_transactions_with_anomalies.csv'") 