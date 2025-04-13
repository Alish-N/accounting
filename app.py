from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import sys
import os
from contextlib import redirect_stdout
import json
import uvicorn
from datetime import datetime, timedelta
import random
import uuid

# Import the anomaly detection class
from datacleaning1.anomaly_detection import TransactionAnomalyDetector

app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="API for detecting anomalies in transaction data",
    version="1.0.0"
)

# Create directories if they don't exist
os.makedirs('cleaned_data', exist_ok=True)
os.makedirs('anomaly_results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('evaluation_results', exist_ok=True)

# Pydantic models for request validation
class Transaction(BaseModel):
    transaction_id: str
    date: str
    transaction_type: Optional[str] = None
    account_type: Optional[str] = None
    amount: float
    currency: Optional[str] = None
    payment_method: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    party_name: Optional[str] = None
    gst_number: Optional[str] = None
    status: Optional[str] = None
    created_by: Optional[str] = None
    approver: Optional[str] = None
    total_tax: Optional[float] = None
    total_tax_calculated: Optional[float] = None
    tax_discrepancy: Optional[float] = None

class TransactionList(BaseModel):
    transactions: List[Transaction]

class User(BaseModel):
    user_id: str
    joining_date: Optional[str] = None
    department: Optional[str] = None

class UserList(BaseModel):
    users: List[User]

# Class to capture console output
class OutputCapture:
    def __init__(self):
        self.output = []
        
    def write(self, text):
        self.output.append(text)
        
    def flush(self):
        pass
        
    def get_output(self):
        return ''.join(self.output)

# Global detector instance
detector = None

@app.get("/")
async def root():
    return {"message": "Transaction Anomaly Detection API", "status": "active"}

@app.post("/api/upload-transactions")
async def upload_transactions(transactions: TransactionList):
    """Upload transaction data for anomaly detection"""
    try:
        # Convert to dataframe
        transactions_df = pd.DataFrame([t.dict() for t in transactions.transactions])
        
        # Convert date strings to datetime
        if 'date' in transactions_df.columns:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
            
        # Save to CSV
        transactions_df.to_csv('cleaned_data/cleaned_transactions.csv', index=False)
        
        return {
            "status": "success", 
            "message": f"Successfully uploaded {len(transactions_df)} transactions",
            "transaction_count": len(transactions_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading transactions: {str(e)}")

@app.post("/api/upload-users")
async def upload_users(users: UserList):
    """Upload user data for behavioral analysis"""
    try:
        # Convert to dataframe
        users_df = pd.DataFrame([u.dict() for u in users.users])
            
        # Save to CSV
        users_df.to_csv('cleaned_data/cleaned_users.csv', index=False)
        
        return {
            "status": "success", 
            "message": f"Successfully uploaded {len(users_df)} users",
            "user_count": len(users_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading users: {str(e)}")

def generate_sample_transactions(num_transactions=100):
    """Generate sample transaction data for testing"""
    transactions = []
    
    # Transaction types and payment methods
    tx_types = ['Purchase', 'Sale', 'Refund', 'Payment', 'Transfer']
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Bank Transfer', 'Check']
    currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD']
    categories = ['Office Supplies', 'Travel', 'Utilities', 'Marketing', 'Equipment']
    statuses = ['Completed', 'Pending', 'Rejected']
    user_ids = [f'user_{i}' for i in range(1, 6)]
    approvers = [f'approver_{i}' for i in range(1, 4)]
    
    # Base date for transactions
    base_date = datetime.now() - timedelta(days=30)
    
    for i in range(num_transactions):
        # Generate random transaction data
        amount = round(random.uniform(10, 5000), 2)
        tax_rate = random.uniform(0.05, 0.2)  # 5% to 20%
        total_tax = round(amount * tax_rate, 2)
        
        # Add a small discrepancy to some transactions to create anomalies
        if random.random() < 0.1:  # 10% chance of an anomaly
            tax_discrepancy = round(random.uniform(10, 100), 2)
            total_tax_calculated = total_tax + tax_discrepancy
        else:
            tax_discrepancy = 0
            total_tax_calculated = total_tax
        
        # Create transaction
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'date': (base_date + timedelta(days=random.randint(0, 30), 
                                          hours=random.randint(0, 23), 
                                          minutes=random.randint(0, 59))).isoformat(),
            'transaction_type': random.choice(tx_types),
            'account_type': 'Business' if random.random() < 0.8 else 'Personal',
            'amount': amount,
            'currency': random.choice(currencies),
            'payment_method': random.choice(payment_methods),
            'category': random.choice(categories),
            'description': f'Sample transaction {i+1}',
            'party_name': f'Vendor {random.randint(1, 20)}',
            'gst_number': f'GST{random.randint(10000, 99999)}',
            'status': random.choice(statuses),
            'created_by': random.choice(user_ids),
            'approver': random.choice(approvers),
            'total_tax': total_tax,
            'total_tax_calculated': total_tax_calculated,
            'tax_discrepancy': tax_discrepancy
        }
        
        transactions.append(transaction)
    
    return transactions

def generate_sample_users():
    """Generate sample user data for testing"""
    users = []
    departments = ['Finance', 'Marketing', 'Sales', 'IT', 'HR']
    
    for i in range(1, 6):
        user = {
            'user_id': f'user_{i}',
            'joining_date': (datetime.now() - timedelta(days=random.randint(30, 1000))).isoformat(),
            'department': random.choice(departments)
        }
        users.append(user)
    
    return users

def ensure_sample_data_exists():
    """Ensure sample data exists for testing"""
    # Create directories if they don't exist
    os.makedirs('cleaned_data', exist_ok=True)
    
    # Check if transaction data exists
    if not os.path.exists('cleaned_data/cleaned_transactions.csv'):
        # Generate sample transactions
        transactions = generate_sample_transactions()
        transactions_df = pd.DataFrame(transactions)
        
        # Convert date strings to datetime
        if 'date' in transactions_df.columns:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
            
        # Save to CSV
        transactions_df.to_csv('cleaned_data/cleaned_transactions.csv', index=False)
        print(f"Generated {len(transactions_df)} sample transactions for testing")
    
    # Check if user data exists
    if not os.path.exists('cleaned_data/cleaned_users.csv'):
        # Generate sample users
        users = generate_sample_users()
        users_df = pd.DataFrame(users)
        
        # Save to CSV
        users_df.to_csv('cleaned_data/cleaned_users.csv', index=False)
        print(f"Generated {len(users_df)} sample users for testing")

@app.post("/api/detect-anomalies")
async def detect_anomalies(background_tasks: BackgroundTasks):
    """Run anomaly detection on the uploaded data"""
    try:
        # Ensure sample data exists for testing
        ensure_sample_data_exists()
        
        # Check if transaction data exists
        if not os.path.exists('cleaned_data/cleaned_transactions.csv'):
            raise HTTPException(status_code=400, detail="No transaction data uploaded. Upload data first.")
        
        # Capture console output
        output_capture = OutputCapture()
        original_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            # Initialize and run detector
            detector = TransactionAnomalyDetector()
            results = detector.detect_anomalies()
            
            # Get anomaly results
            anomaly_df = results[results['is_anomaly']]
            
            # Prepare response data
            response_data = {
                "status": "success",
                "total_transactions": len(results),
                "anomaly_count": len(anomaly_df),
                "anomaly_percentage": (len(anomaly_df) / len(results) * 100) if len(results) > 0 else 0,
                "console_output": output_capture.get_output(),
                "anomalies": []
            }
            
            # Add sample anomalies
            if len(anomaly_df) > 0:
                # Select only the relevant columns for the response
                display_columns = [
                    'transaction_id', 'date', 'amount', 'tax_discrepancy',
                    'is_anomaly', 'anomaly_score'
                ]
                display_columns = [col for col in display_columns if col in anomaly_df.columns]
                
                # Convert sample anomalies to list of dicts (handle datetime serialization)
                sample_anomalies = anomaly_df[display_columns].head(10).to_dict('records')
                
                # Convert datetime to string
                for anomaly in sample_anomalies:
                    if 'date' in anomaly and isinstance(anomaly['date'], datetime):
                        anomaly['date'] = anomaly['date'].isoformat()
                
                response_data['anomalies'] = sample_anomalies
                
            return JSONResponse(content=response_data)
            
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during anomaly detection: {str(e)}")

@app.get("/api/anomaly-results")
async def get_anomaly_results():
    """Get the results of the last anomaly detection run"""
    try:
        # Check if results file exists
        results_path = 'anomaly_results/complete_transactions_with_anomalies.csv'
        metrics_path = 'evaluation_results/anomaly_detection_metrics.csv'
        
        # If no results exist, suggest running the detection
        if not os.path.exists(results_path):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "No anomaly detection results found. Please run detection first by accessing /api/detect-anomalies endpoint.",
                    "suggestion": "You can run anomaly detection on sample data by making a POST request to /api/detect-anomalies"
                }
            )
        
        # Load results
        results_df = pd.read_csv(results_path)
        anomaly_df = results_df[results_df['is_anomaly']]
        
        # Load metrics if available
        metrics = {}
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            if not metrics_df.empty:
                metrics = metrics_df.iloc[0].to_dict()
        
        # Prepare response
        response_data = {
            "status": "success",
            "total_transactions": len(results_df),
            "anomaly_count": len(anomaly_df),
            "anomaly_percentage": (len(anomaly_df) / len(results_df) * 100) if len(results_df) > 0 else 0,
            "metrics": metrics,
            "anomalies": []
        }
        
        # Add anomalies to response
        if len(anomaly_df) > 0:
            # Select only the relevant columns for the response
            display_columns = [
                'transaction_id', 'date', 'amount', 'tax_discrepancy',
                'is_anomaly', 'anomaly_score'
            ]
            display_columns = [col for col in display_columns if col in anomaly_df.columns]
            
            # Convert anomalies to list of dicts
            anomalies = anomaly_df[display_columns].to_dict('records')
            response_data['anomalies'] = anomalies
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving anomaly results: {str(e)}")

# Add this new test endpoint to generate sample data and run detection
@app.post("/api/test")
async def generate_test_data_and_detect():
    """Generate sample data and run anomaly detection for testing"""
    try:
        # Generate sample data
        ensure_sample_data_exists()
        
        # Capture console output
        output_capture = OutputCapture()
        original_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            # Initialize and run detector
            detector = TransactionAnomalyDetector()
            results = detector.detect_anomalies()
            
            # Get anomaly results
            anomaly_df = results[results['is_anomaly']]
            
            # Prepare response data
            response_data = {
                "status": "success",
                "message": "Sample data generated and anomaly detection completed successfully",
                "total_transactions": len(results),
                "anomaly_count": len(anomaly_df),
                "anomaly_percentage": (len(anomaly_df) / len(results) * 100) if len(results) > 0 else 0,
                "console_output": output_capture.get_output()
            }
            
            return JSONResponse(content=response_data)
            
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during test: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
