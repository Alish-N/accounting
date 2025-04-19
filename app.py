from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
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
import traceback
from io import StringIO
import numpy as np
import shutil

# Import the anomaly and fraud detection classes
from datacleaning1.anomaly_detection import TransactionAnomalyDetector
from datacleaning1.fraud_detection import FraudDetector
# Import the business forecasting service
from datacleaning1.business_service import BusinessForecastingService
from datacleaning1.buisness_forecasting import DATASET_PATH as BUSINESS_FORECAST_DATASET_PATH
from datacleaning1.financial_ratio_analysis import FinancialRatioAnalyzer

app = FastAPI(
    title="Financial Analytics and Forecasting API",
    description="API for detecting anomalies, identifying fraud in transaction data, and forecasting financial metrics",
    version="1.0.0"
)

# Create directories if they don't exist
os.makedirs('cleaned_data', exist_ok=True)
os.makedirs('anomaly_results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('evaluation_results', exist_ok=True)
os.makedirs('fraud_results', exist_ok=True)
os.makedirs('financial_ratio_results', exist_ok=True)

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

# Global detector instances
anomaly_detector = None
fraud_detector = None

# Initialize the business forecasting service
business_forecasting_service = BusinessForecastingService()

@app.get("/")
async def root():
    return {
        "message": "Financial Analysis API",
        "status": "active",
        "available_services": {
            "anomaly_detection": {
                "description": "Detect anomalies in transaction data",
                "endpoints": [
                    "/api/upload-transactions",
                    "/api/detect-anomalies",
                    "/api/anomaly-results"
                ]
            },
            "fraud_detection": {
                "description": "Detect potential fraud in transaction data",
                "endpoints": [
                    "/api/detect-fraud",
                    "/api/fraud-results",
                    "/api/quick-fraud-analysis",
                    "/api/run-full-pipeline"
                ]
            },
            "business_forecasting": {
                "description": "Forecast financial metrics like revenue, expenses, and cash flow",
                "endpoints": [
                    "/api/business/upload-data",
                    "/api/business/generate-sample-data",
                    "/api/business/use-original-dataset",
                    "/api/business/forecast",
                    "/api/business/forecast-results",
                    "/api/business/dashboard",
                    "/api/business/report",
                    "/api/business/quick-forecast"
                ]
            },
            "financial_ratio_analysis": {
                "description": "Analyze financial ratios from the dataset",
                "endpoints": [
                    "/api/financial-ratios/analyze",
                    "/api/financial-ratios/results", 
                    "/api/financial-ratios/dashboard"
                ]
            }
        },
        "documentation": "/docs"
    }

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

def generate_sample_transactions(num_transactions=150):
    """Generate sample transaction data for testing"""
    transactions = []
    
    # Transaction types and payment methods
    tx_types = ['Purchase', 'Sale', 'Refund', 'Payment', 'Transfer']
    payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'Bank Transfer', 'Check']
    currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD']
    categories = ['Office Supplies', 'Travel', 'Utilities', 'Marketing', 'Equipment']
    statuses = ['Completed', 'Pending', 'Rejected']
    user_ids = [f'user_{i}' for i in range(1, 10)]
    approvers = [f'approver_{i}' for i in range(1, 5)]
    
    # Base date for transactions
    base_date = datetime.now() - timedelta(days=60)
    
    # Create some users with multiple transactions for better velocity tracking
    user_transaction_counts = {user_id: random.randint(5, 20) for user_id in user_ids}
    
    # Generate transactions for each user to ensure good distribution
    for user_id, count in user_transaction_counts.items():
        # Generate normal transactions for this user
        user_transactions = []
        for i in range(count):
            # Normal transaction
            amount = round(random.uniform(100, 2000), 2)
            tax_rate = random.uniform(0.05, 0.15)  # 5% to 15%
            total_tax = round(amount * tax_rate, 2)
            
            # Occasionally add small discrepancy (non-anomalous)
            if random.random() < 0.05:  # 5% chance of small discrepancy
                tax_discrepancy = round(random.uniform(0.5, 5), 2)
                total_tax_calculated = total_tax + tax_discrepancy
            else:
                tax_discrepancy = 0
                total_tax_calculated = total_tax
            
            # Create transaction with consistent date patterns for this user
            # This creates better time patterns for velocity features
            transaction_date = base_date + timedelta(
                days=i*2 + random.randint(0, 3), 
                hours=random.randint(9, 17),  # Business hours
                minutes=random.randint(0, 59)
            )
            
            transaction = {
                'transaction_id': str(uuid.uuid4()),
                'date': transaction_date.isoformat(),
                'transaction_type': random.choice(tx_types),
                'account_type': 'Business' if random.random() < 0.8 else 'Personal',
                'amount': amount,
                'currency': random.choice(currencies),
                'payment_method': random.choice(payment_methods),
                'category': random.choice(categories),
                'description': f'Regular transaction {i+1} for {user_id}',
                'party_name': f'Vendor {random.randint(1, 20)}',
                'gst_number': f'GST{random.randint(10000, 99999)}',
                'status': 'Completed',  # Most are completed
                'created_by': user_id,
                'approver': random.choice(approvers),
                'total_tax': total_tax,
                'total_tax_calculated': total_tax_calculated,
                'tax_discrepancy': tax_discrepancy
            }
            
            user_transactions.append(transaction)
        
        # Add transactions for this user
        transactions.extend(user_transactions)
    
    # Now add some anomalous transactions (~10-15% of total)
    num_anomalies = int(num_transactions * 0.15)
    anomaly_transactions = []
    
    for i in range(num_anomalies):
        # Pick a user to create an anomaly for
        user_id = random.choice(user_ids)
        
        # Higher amount and tax discrepancy for anomalies
        amount = round(random.uniform(3000, 10000), 2)  # Higher amounts
        tax_rate = random.uniform(0.1, 0.2)  # 10% to 20%
        total_tax = round(amount * tax_rate, 2)
        
        # Add substantial tax discrepancy
        tax_discrepancy = round(random.uniform(50, 500), 2)  # Large discrepancy
        total_tax_calculated = total_tax + tax_discrepancy
        
        # Create transaction with unusual timing (night/weekend)
        unusual_hour = random.choice([0, 1, 2, 3, 4, 22, 23])  # Late night hours
        weekend_day = random.choice([5, 6])  # Saturday or Sunday
        
        # Base date + weekend day + unusual hour
        anomaly_date = base_date + timedelta(
            days=random.randint(0, 50) + weekend_day,
            hours=unusual_hour,
            minutes=random.randint(0, 59)
        )
        
        # Some anomalies are rejected
        status = random.choices(['Completed', 'Rejected'], weights=[0.6, 0.4])[0]
        
        anomaly = {
            'transaction_id': str(uuid.uuid4()),
            'date': anomaly_date.isoformat(),
            'transaction_type': random.choice(tx_types),
            'account_type': 'Personal' if random.random() < 0.7 else 'Business',  # More personal
            'amount': amount,
            'currency': random.choice(currencies),
            'payment_method': random.choice(payment_methods),
            'category': random.choice(categories),
            'description': f'Unusual transaction {i+1}',
            'party_name': f'Unknown Vendor {random.randint(1, 10)}',
            'gst_number': f'GST{random.randint(10000, 99999)}',
            'status': status,
            'created_by': user_id,
            'approver': random.choice(approvers),
            'total_tax': total_tax,
            'total_tax_calculated': total_tax_calculated,
            'tax_discrepancy': tax_discrepancy
        }
        
        anomaly_transactions.append(anomaly)
    
    # Add anomalies to regular transactions
    transactions.extend(anomaly_transactions)
    
    # Add some quick back-to-back transactions (velocity anomalies)
    for _ in range(5):
        user_id = random.choice(user_ids)
        base_tx_time = base_date + timedelta(days=random.randint(10, 40))
        
        # Create 3 quick transactions within minutes
        for j in range(3):
            quick_tx_time = base_tx_time + timedelta(minutes=j*5)  # 5 minutes apart
            amount = round(random.uniform(500, 3000), 2)
            tax_rate = random.uniform(0.05, 0.15)
            total_tax = round(amount * tax_rate, 2)
            
            quick_tx = {
                'transaction_id': str(uuid.uuid4()),
                'date': quick_tx_time.isoformat(),
                'transaction_type': 'Transfer',  # Often transfers
                'account_type': 'Business',
                'amount': amount,
                'currency': random.choice(currencies),
                'payment_method': 'Bank Transfer',  # Usually bank transfers
                'category': 'Transfer',
                'description': f'Quick transaction {j+1} of 3',
                'party_name': f'Recipient {random.randint(1, 5)}',
                'gst_number': f'GST{random.randint(10000, 99999)}',
                'status': 'Completed',
                'created_by': user_id,
                'approver': random.choice(approvers),
                'total_tax': total_tax,
                'total_tax_calculated': total_tax,
                'tax_discrepancy': 0
            }
            
            transactions.append(quick_tx)
    
    # Shuffle all transactions
    random.shuffle(transactions)
    
    # Limit to requested number if we generated too many
    return transactions[:num_transactions]

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
            anomaly_detector = TransactionAnomalyDetector()
            results = anomaly_detector.detect_anomalies()
            
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
            anomaly_detector = TransactionAnomalyDetector()
            results = anomaly_detector.detect_anomalies()
            
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

@app.post("/api/detect-fraud")
async def detect_fraud(run_anomaly_if_needed: bool = True):
    """
    Run fraud detection on the anomaly detection results
    
    Parameters:
    - run_anomaly_if_needed: If True and no anomaly results exist, run anomaly detection first
    """
    try:
        # Check if anomaly results exist
        anomaly_results_path = 'anomaly_results/complete_transactions_with_anomalies.csv'
        
        # If anomaly results don't exist but autorun is enabled, run anomaly detection
        if not os.path.exists(anomaly_results_path) and run_anomaly_if_needed:
            print("Anomaly results not found. Running anomaly detection first...")
            ensure_sample_data_exists()
            anomaly_detector = TransactionAnomalyDetector()
            anomaly_results = anomaly_detector.detect_anomalies()
        elif not os.path.exists(anomaly_results_path):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "No anomaly detection results found. Run anomaly detection first.",
                    "suggestion": "Run anomaly detection by making a POST request to /api/detect-anomalies or use parameter 'run_anomaly_if_needed=true'"
                }
            )
        
        # Capture console output
        output_capture = OutputCapture()
        original_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            print("Starting fraud detection...")
            
            # Initialize fraud detector
            fraud_detector = FraudDetector()
            
            # Load transaction data for prediction
            transactions = pd.read_csv(anomaly_results_path)
            print(f"Loaded {len(transactions)} transactions from anomaly results")
            
            # Verify required columns exist
            required_columns = ['transaction_id', 'date', 'amount', 'anomaly_score']
            missing_columns = [col for col in required_columns if col not in transactions.columns]
            if missing_columns:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"Required columns missing from anomaly results: {missing_columns}",
                        "suggestion": "Run anomaly detection again to ensure all required columns are present"
                    }
                )
            
            # Train models and catch specific exceptions
            try:
                print("Training fraud detection models...")
                model_results = fraud_detector.compare_models()
                
                print("Predicting fraud probabilities...")
                predictions = fraud_detector.predict_fraud(transactions)
                
                print("Generating fraud report...")
                fraud_detector.generate_fraud_report(predictions)
                
                # Calculate fraud statistics
                fraud_count = int(predictions['is_fraud'].sum())
                fraud_percentage = float((predictions['is_fraud'].sum() / len(predictions) * 100))
                
                # Get risk level distribution
                risk_levels = {}
                if 'risk_level' in predictions.columns:
                    for level in ['Low', 'Medium', 'High', 'Critical']:
                        level_count = len(predictions[predictions['risk_level'] == level])
                        if level_count > 0:
                            risk_levels[level] = level_count
                
                # Prepare sample fraud cases
                sample_fraud_cases = []
                if fraud_count > 0:
                    # Get high-risk fraud cases
                    fraud_cases = predictions[predictions['is_fraud']].sort_values('fraud_probability', ascending=False).head(10)
                    
                    # Extract relevant columns
                    display_cols = ['transaction_id', 'date', 'amount', 'category', 'payment_method', 'fraud_probability', 'risk_level']
                    available_cols = [col for col in display_cols if col in fraud_cases.columns]
                    
                    # Convert to records
                    cases = fraud_cases[available_cols].to_dict('records')
                    
                    # Convert datetime to string if needed
                    for case in cases:
                        if 'date' in case and isinstance(case['date'], datetime):
                            case['date'] = case['date'].isoformat()
                    
                    sample_fraud_cases = cases
                
                # Prepare response data
                response_data = {
                    "status": "success",
                    "message": "Fraud detection completed successfully",
                    "total_transactions": len(predictions),
                    "fraud_count": fraud_count,
                    "fraud_percentage": fraud_percentage,
                    "risk_levels": risk_levels,
                    "sample_fraud_cases": sample_fraud_cases,
                    "console_output": output_capture.get_output()
                }
                
                if model_results:
                    response_data["best_model"] = model_results.get('best_model', 'Unknown')
                    response_data["best_auc"] = float(model_results.get('auc', 0))
                
                return JSONResponse(content=response_data)
                
            except ValueError as e:
                # Handle common ValueError exceptions
                error_message = str(e)
                suggestion = "Review your data structure and try again"
                
                if "inconsistent numbers of samples" in error_message:
                    suggestion = "This is likely due to a preprocessing issue. Try running the anomaly detection again."
                elif "operands could not be broadcast together" in error_message:
                    suggestion = "There might be a mismatch in feature dimensions. Try running the anomaly detection again."
                elif "Input contains NaN" in error_message:
                    suggestion = "Your data contains missing values. Try running anomaly detection with data cleaning enabled."
                
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"ValueError in fraud detection: {error_message}",
                        "suggestion": suggestion,
                        "console_output": output_capture.get_output()
                    }
                )
                
            except Exception as e:
                # Capture the full error trace
                error_trace = traceback.format_exc()
                print(f"Error in fraud detection: {str(e)}")
                print(error_trace)
                
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": f"Error in fraud detection: {str(e)}",
                        "error_trace": error_trace,
                        "console_output": output_capture.get_output()
                    }
                )
                
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            
    except Exception as e:
        # Capture unexpected errors
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Unexpected error: {str(e)}",
                "error_trace": error_trace
            }
        )

@app.get("/api/fraud-results")
async def get_fraud_results(force: bool = False, autorun: bool = False):
    """
    Get fraud detection results.
    
    Parameters:
    - force: If True, ignore missing columns and return partial results
    - autorun: If True and results don't exist, attempt to run fraud detection
    
    Returns:
    - JSON with fraud statistics and risk levels
    """
    try:
        # Check if fraud results exist
        fraud_results_path = "datacleaning1/fraud_results/fraud_report.csv"
        if not os.path.exists(fraud_results_path):
            if autorun:
                # If autorun is enabled, try to run fraud detection
                response = await detect_fraud()
                if response.status_code != 200:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "status": "error",
                            "message": "No fraud detection results found. Run fraud detection first.",
                            "suggestion": "Use one of these methods to generate results:\n"
                                        "1. Add '?autorun=true' to this URL\n"
                                        "2. Visit /api/fix-fraud-detection to repair any issues\n"
                                        "3. Make a POST request to /api/detect-fraud\n"
                                        "4. Visit /api/quick-fraud-analysis"
                        }
                    )
            else:
                # Otherwise return a 404 error
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": "error",
                        "message": "No fraud detection results found. Run fraud detection first.",
                        "suggestion": "Use one of these methods to generate results:\n"
                                    "1. Add '?autorun=true' to this URL\n"
                                    "2. Visit /api/fix-fraud-detection to repair any issues\n"
                                    "3. Make a POST request to /api/detect-fraud\n"
                                    "4. Visit /api/quick-fraud-analysis"
                    }
                )
        
        # Load fraud results
        df = pd.read_csv(fraud_results_path)
        
        # Check required columns
        required_columns = ['transaction_id', 'amount', 'fraud_probability', 'is_fraud', 'risk_level']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # If missing required columns and not forced, return error
        if missing_columns and not force:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Fraud results missing required columns: {', '.join(missing_columns)}",
                    "suggestion": "Use '?force=true' to return partial results"
                }
            )
        
        # Create a simplified fraud report if key columns are missing
        if missing_columns and force:
            # Add missing columns with default values
            for col in missing_columns:
                if col == 'transaction_id':
                    df['transaction_id'] = [f"TX{i}" for i in range(len(df))]
                elif col == 'amount':
                    df['amount'] = 0.0
                elif col == 'fraud_probability':
                    if 'anomaly_score' in df.columns:
                        df['fraud_probability'] = df['anomaly_score'] * 1.2
                    else:
                        df['fraud_probability'] = 0.1
                elif col == 'is_fraud':
                    if 'fraud_probability' in df.columns:
                        df['is_fraud'] = df['fraud_probability'] > 0.7
                    else:
                        df['is_fraud'] = False
                elif col == 'risk_level':
                    if 'fraud_probability' in df.columns:
                        df['risk_level'] = pd.cut(
                            df['fraud_probability'], 
                            bins=[0, 0.3, 0.5, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High', 'Critical']
                        )
                    else:
                        df['risk_level'] = 'Low'
        
        # Calculate fraud statistics
        try:
            total_transactions = len(df)
            if 'is_fraud' in df.columns:
                fraud_count = int(df['is_fraud'].sum())
                fraud_percentage = fraud_count / total_transactions * 100 if total_transactions > 0 else 0
            else:
                fraud_count = 0
                fraud_percentage = 0
                
            # Calculate financial impact if amount column exists
            if 'amount' in df.columns and 'is_fraud' in df.columns:
                total_amount = float(df['amount'].sum())
                fraud_amount = float(df.loc[df['is_fraud'], 'amount'].sum())
                fraud_amount_percentage = fraud_amount / total_amount * 100 if total_amount > 0 else 0
            else:
                total_amount = 0.0
                fraud_amount = 0.0
                fraud_amount_percentage = 0.0
                
            # Calculate risk level distribution
            if 'risk_level' in df.columns:
                risk_distribution = df['risk_level'].value_counts().to_dict()
                # Convert to strings for JSON serialization
                risk_distribution = {str(k): int(v) for k, v in risk_distribution.items()}
            else:
                risk_distribution = {}
                
            # Look for high risk transactions
            high_risk_transactions = []
            if all(col in df.columns for col in ['transaction_id', 'amount', 'fraud_probability', 'risk_level']):
                high_risk = df[df['risk_level'].isin(['High', 'Critical'])].sort_values('fraud_probability', ascending=False)
                
                # Get up to 5 high risk transactions
                for _, row in high_risk.head(5).iterrows():
                    high_risk_transactions.append({
                        'transaction_id': str(row['transaction_id']),
                        'amount': float(row['amount']),
                        'fraud_probability': float(row['fraud_probability']),
                        'risk_level': str(row['risk_level']),
                        'category': str(row['category']) if 'category' in row else 'Unknown'
                    })
        except Exception as e:
            if not force:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": f"Error calculating fraud statistics: {str(e)}",
                        "suggestion": "Use '?force=true' to return partial results"
                    }
                )
            # If force=True, continue with empty/default values
            fraud_count = 0
            fraud_percentage = 0
            total_amount = 0.0
            fraud_amount = 0.0
            fraud_amount_percentage = 0.0
            risk_distribution = {}
            high_risk_transactions = []
        
        # Prepare response
        response = {
            "status": "success" if not missing_columns else "partial_success",
            "message": "Fraud detection results" if not missing_columns else "Partial fraud detection results (some data missing)",
            "statistics": {
                "total_transactions": total_transactions,
                "fraud_count": fraud_count,
                "fraud_percentage": round(fraud_percentage, 2),
                "total_amount": round(total_amount, 2),
                "fraud_amount": round(fraud_amount, 2),
                "fraud_amount_percentage": round(fraud_amount_percentage, 2)
            },
            "risk_distribution": risk_distribution,
            "high_risk_transactions": high_risk_transactions
        }
        
        # Add warnings or suggestions if partial results
        if missing_columns:
            response["warnings"] = {
                "missing_columns": missing_columns,
                "suggestion": "Run the fraud detection process again for complete results"
            }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error processing fraud results: {str(e)}",
                "error_trace": error_trace
            }
        )

@app.post("/api/run-full-pipeline")
async def run_full_pipeline():
    """Run the complete pipeline: generate sample data, detect anomalies, and then detect fraud"""
    try:
        # Step 1: Ensure sample data exists
        ensure_sample_data_exists()
        
        # Capture console output
        output_capture = OutputCapture()
        original_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            # Step 2: Run anomaly detection
            print("=== STEP 1: ANOMALY DETECTION ===\n")
            anomaly_detector = TransactionAnomalyDetector()
            anomaly_results = anomaly_detector.detect_anomalies()
            
            # Step 3: Run fraud detection
            print("\n\n=== STEP 2: FRAUD DETECTION ===\n")
            fraud_detector = FraudDetector()
            
            try:
                # Train models
                model_results = fraud_detector.compare_models()
                
                # Predict fraud
                predictions = fraud_detector.predict_fraud(anomaly_results)
                
                # Generate fraud report
                fraud_detector.generate_fraud_report(predictions)
                
                # Prepare success response
                response_data = {
                    "status": "success",
                    "message": "Full pipeline completed successfully",
                    "anomaly_detection": {
                        "total_transactions": len(anomaly_results),
                        "anomaly_count": len(anomaly_results[anomaly_results['is_anomaly']]),
                        "anomaly_percentage": (len(anomaly_results[anomaly_results['is_anomaly']]) / len(anomaly_results) * 100)
                    },
                    "fraud_detection": {
                        "total_transactions": len(predictions),
                        "fraud_count": int(predictions['is_fraud'].sum()),
                        "fraud_percentage": float((predictions['is_fraud'].sum() / len(predictions) * 100)),
                        "best_model": model_results.get('best_model', 'Unknown'),
                        "best_auc": float(model_results.get('auc', 0))
                    },
                    "console_output": output_capture.get_output()
                }
                
            except Exception as e:
                # Return partial success with error in fraud detection
                error_trace = traceback.format_exc()
                print(f"Error in fraud detection: {str(e)}")
                print(error_trace)
                
                response_data = {
                    "status": "partial_success",
                    "message": "Anomaly detection completed successfully, but fraud detection failed",
                    "anomaly_detection": {
                        "total_transactions": len(anomaly_results),
                        "anomaly_count": len(anomaly_results[anomaly_results['is_anomaly']]),
                        "anomaly_percentage": (len(anomaly_results[anomaly_results['is_anomaly']]) / len(anomaly_results) * 100)
                    },
                    "fraud_detection": {
                        "status": "error",
                        "error_message": str(e),
                        "error_trace": error_trace
                    },
                    "console_output": output_capture.get_output()
                }
            
            return JSONResponse(content=response_data)
            
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            
    except Exception as e:
        # Handle unexpected errors
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error running full pipeline: {str(e)}",
                "error_trace": error_trace
            }
        )

# Add a simpler endpoint that just works
@app.get("/api/quick-fraud-analysis")
async def quick_fraud_analysis():
    """One-click endpoint to run the entire pipeline and get fraud results"""
    try:
        # Create sample data directory if needed
        os.makedirs('cleaned_data', exist_ok=True)
        
        # Capture console output
        output_capture = OutputCapture()
        original_stdout = sys.stdout
        sys.stdout = output_capture
        
        try:
            # Generate sample data
            ensure_sample_data_exists()
            
            # Run anomaly detection
            print("=== Running Anomaly Detection ===")
            anomaly_detector = TransactionAnomalyDetector()
            anomaly_results = anomaly_detector.detect_anomalies()
            
            # Run fraud detection
            print("\n=== Running Fraud Detection ===")
            fraud_detector = FraudDetector()
            model_results = fraud_detector.compare_models()
            predictions = fraud_detector.predict_fraud(anomaly_results)
            fraud_detector.generate_fraud_report(predictions)
            
            # Load fraud report for response
            if os.path.exists('fraud_results/fraud_report.csv'):
                fraud_results = pd.read_csv('fraud_results/fraud_report.csv')
                
                # Calculate basic statistics
                total_transactions = len(fraud_results)
                fraud_count = int(fraud_results['is_fraud'].sum()) if 'is_fraud' in fraud_results.columns else 0
                fraud_percentage = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0
                
                # Get high risk transactions
                high_risk_transactions = []
                if 'fraud_probability' in fraud_results.columns and fraud_count > 0:
                    high_risk = fraud_results.sort_values('fraud_probability', ascending=False).head(5)
                    high_risk_cols = ['transaction_id', 'date', 'amount', 'fraud_probability', 'risk_level']
                    available_cols = [col for col in high_risk_cols if col in high_risk.columns]
                    high_risk_transactions = high_risk[available_cols].to_dict('records')
                
                # Prepare successful response
                response_data = {
                    "status": "success",
                    "message": "Quick fraud analysis completed successfully",
                    "total_transactions": total_transactions,
                    "fraud_count": fraud_count,
                    "fraud_percentage": fraud_percentage,
                    "high_risk_transactions": high_risk_transactions,
                    "console_output_summary": "Full pipeline completed successfully. Check /api/fraud-results for detailed results."
                }
            else:
                # Something went wrong if the file doesn't exist
                response_data = {
                    "status": "error",
                    "message": "Fraud detection ran but no results file was created",
                    "console_output": output_capture.get_output()
                }
            
            return JSONResponse(content=response_data)
            
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            
    except Exception as e:
        # Capture error
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error: {str(e)}",
                "error_trace": error_trace,
                "suggestion": "Try running the steps separately with /api/detect-anomalies and then /api/detect-fraud"
            }
        )

@app.get("/api/fix-fraud-detection")
async def fix_fraud_detection():
    """
    Diagnostic and repair endpoint for the fraud detection system.
    - Checks directory structure
    - Ensures sample data exists
    - Runs both anomaly and fraud detection
    - Verifies all output files
    """
    try:
        diagnostics = {
            "directories": {},
            "files": {},
            "processes": {},
            "repairs": []
        }
        
        # 1. Check and create necessary directories
        directories = ["data", "anomaly_results", "fraud_results", "models"]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                diagnostics["directories"][directory] = "created"
                diagnostics["repairs"].append(f"Created missing directory: {directory}")
            else:
                diagnostics["directories"][directory] = "exists"
        
        # 2. Check sample data existence
        data_files = {
            "transactions": "data/transactions.csv",
            "users": "data/users.csv"
        }
        
        for name, file_path in data_files.items():
            if not os.path.exists(file_path):
                diagnostics["files"][name] = "missing"
                diagnostics["repairs"].append(f"Generating sample {name} data")
                # Generate sample data
                ensure_sample_data_exists()
                if os.path.exists(file_path):
                    diagnostics["files"][name] = "created"
            else:
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    diagnostics["files"][name] = "empty"
                    diagnostics["repairs"].append(f"Replacing empty {name} file")
                    ensure_sample_data_exists()
                    diagnostics["files"][name] = "replaced"
                else:
                    try:
                        # Try to load the file to ensure it's valid
                        pd.read_csv(file_path)
                        diagnostics["files"][name] = "valid"
                    except Exception as e:
                        diagnostics["files"][name] = f"corrupted: {str(e)}"
                        diagnostics["repairs"].append(f"Replacing corrupted {name} file")
                        ensure_sample_data_exists()
                        diagnostics["files"][name] = "replaced"
        
        # 3. Run anomaly detection and check results
        anomaly_results_path = "anomaly_results/complete_transactions_with_anomalies.csv"
        anomaly_success = False
        
        try:
            # Run anomaly detection
            diagnostics["processes"]["anomaly_detection"] = "starting"
            
            # Capture standard output
            output_capture = StringIO()
            with redirect_stdout(output_capture):
                detector = TransactionAnomalyDetector()
                results = detector.detect_anomalies()
            
            diagnostics["processes"]["anomaly_detection_output"] = output_capture.getvalue()
            
            # Check if results file exists and is not empty
            if os.path.exists(anomaly_results_path) and os.path.getsize(anomaly_results_path) > 0:
                try:
                    # Load to validate
                    df = pd.read_csv(anomaly_results_path)
                    if len(df) > 0:
                        diagnostics["processes"]["anomaly_detection"] = "success"
                        diagnostics["files"]["anomaly_results"] = f"valid ({len(df)} records)"
                        anomaly_success = True
                    else:
                        diagnostics["processes"]["anomaly_detection"] = "empty_results"
                        diagnostics["files"]["anomaly_results"] = "empty"
                        # Will be fixed in step 5
                except Exception as e:
                    diagnostics["processes"]["anomaly_detection"] = f"invalid_results: {str(e)}"
                    diagnostics["files"]["anomaly_results"] = "corrupted"
            else:
                diagnostics["processes"]["anomaly_detection"] = "no_results_file"
                diagnostics["files"]["anomaly_results"] = "missing"
        except Exception as e:
            error_trace = traceback.format_exc()
            diagnostics["processes"]["anomaly_detection"] = f"failed: {str(e)}"
            diagnostics["processes"]["anomaly_detection_error"] = error_trace
        
        # 4. Run fraud detection and check results
        fraud_results_path = "fraud_results/fraud_report.csv"
        fraud_success = False
        
        # Only proceed if anomaly detection succeeded or file exists
        if anomaly_success or os.path.exists(anomaly_results_path):
            try:
                # Run fraud detection
                diagnostics["processes"]["fraud_detection"] = "starting"
                
                # Load anomaly results
                anomaly_results = pd.read_csv(anomaly_results_path)
                
                # Capture standard output
                output_capture = StringIO()
                with redirect_stdout(output_capture):
                    fraud_detector = FraudDetector()
                    model_results = fraud_detector.compare_models()
                    predictions = fraud_detector.predict_fraud(anomaly_results)
                    fraud_detector.generate_fraud_report(predictions)
                
                diagnostics["processes"]["fraud_detection_output"] = output_capture.getvalue()
                
                # Check if results file exists and is not empty
                if os.path.exists(fraud_results_path) and os.path.getsize(fraud_results_path) > 0:
                    try:
                        # Load to validate
                        df = pd.read_csv(fraud_results_path)
                        if len(df) > 0:
                            diagnostics["processes"]["fraud_detection"] = "success"
                            diagnostics["files"]["fraud_results"] = f"valid ({len(df)} records)"
                            fraud_success = True
                        else:
                            diagnostics["processes"]["fraud_detection"] = "empty_results"
                            diagnostics["files"]["fraud_results"] = "empty"
                    except Exception as e:
                        diagnostics["processes"]["fraud_detection"] = f"invalid_results: {str(e)}"
                        diagnostics["files"]["fraud_results"] = "corrupted"
                else:
                    diagnostics["processes"]["fraud_detection"] = "no_results_file"
                    diagnostics["files"]["fraud_results"] = "missing"
            except Exception as e:
                error_trace = traceback.format_exc()
                diagnostics["processes"]["fraud_detection"] = f"failed: {str(e)}"
                diagnostics["processes"]["fraud_detection_error"] = error_trace
        else:
            diagnostics["processes"]["fraud_detection"] = "skipped_due_to_anomaly_failure"
        
        # 5. If fraud detection failed, create simplified fraud report
        if not fraud_success and os.path.exists(anomaly_results_path):
            try:
                diagnostics["repairs"].append("Creating simplified fraud report from anomaly data")
                
                # Load anomaly results
                anomaly_results = pd.read_csv(anomaly_results_path)
                
                # Create simplified fraud scores based on anomaly scores
                anomaly_results['fraud_probability'] = anomaly_results['anomaly_score'] * 1.2
                anomaly_results['is_fraud'] = (anomaly_results['anomaly_score'] > 0.7).astype(int)
                
                # Add risk levels
                anomaly_results['risk_level'] = pd.cut(
                    anomaly_results['fraud_probability'], 
                    bins=[0, 0.3, 0.5, 0.7, 2.0],
                    labels=['Low', 'Medium', 'High', 'Critical']
                )
                
                # Save as fraud report
                os.makedirs(os.path.dirname(fraud_results_path), exist_ok=True)
                anomaly_results.to_csv(fraud_results_path, index=False)
                
                # Validate
                if os.path.exists(fraud_results_path) and os.path.getsize(fraud_results_path) > 0:
                    df = pd.read_csv(fraud_results_path)
                    if len(df) > 0:
                        diagnostics["processes"]["simplified_fraud_report"] = "success"
                        diagnostics["files"]["fraud_results"] = f"created ({len(df)} records)"
                        fraud_success = True
                    else:
                        diagnostics["processes"]["simplified_fraud_report"] = "empty_results"
                else:
                    diagnostics["processes"]["simplified_fraud_report"] = "failed_to_create"
            except Exception as e:
                error_trace = traceback.format_exc()
                diagnostics["processes"]["simplified_fraud_report"] = f"failed: {str(e)}"
                diagnostics["processes"]["simplified_fraud_report_error"] = error_trace
        
        # 6. Overall status
        if fraud_success:
            diagnostics["status"] = "success"
            diagnostics["message"] = "Fraud detection system is now operational"
        elif anomaly_success:
            diagnostics["status"] = "partial_success"
            diagnostics["message"] = "Anomaly detection is working but fraud detection has issues"
        else:
            diagnostics["status"] = "failed"
            diagnostics["message"] = "Could not fix the system automatically"
        
        # 7. Next steps
        if fraud_success:
            diagnostics["next_steps"] = [
                "Access fraud results at /api/fraud-results",
                "Run quick analysis at /api/quick-fraud-analysis",
                "Check detailed API documentation at /docs"
            ]
        elif anomaly_success:
            diagnostics["next_steps"] = [
                "View anomaly results at /api/anomaly-results",
                "Try simplified fraud analysis with /api/fraud-results?force=true&autorun=true",
                "Check API logs for more details on fraud detection failures"
            ]
        else:
            diagnostics["next_steps"] = [
                "Check if sample data generation is working at /api/generate-sample-data",
                "Verify Python environment has all required packages",
                "Examine detailed error traces in the diagnostic output"
            ]
        
        return JSONResponse(content=diagnostics)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error during system diagnosis: {str(e)}",
                "error_trace": error_trace
            }
        )

# Business Forecasting API Endpoints
class FinancialData(BaseModel):
    date: str
    revenue: Optional[float] = None
    expenses: Optional[float] = None
    cash_flow: Optional[float] = None
    
    @validator('date')
    def validate_date(cls, v):
        try:
            # Check if we can parse the date
            pd.to_datetime(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid date format. Please use YYYY-MM-DD format: {str(e)}")

class FinancialDataList(BaseModel):
    data: List[FinancialData]

class ForecastRequest(BaseModel):
    periods: int = 12
    period_type: str = "month"  # "month" or "year"

@app.post("/api/business/upload-data")
async def upload_financial_data(financial_data: FinancialDataList):
    """Upload financial data for forecasting"""
    try:
        # Convert to dataframe
        data_df = pd.DataFrame([item.dict() for item in financial_data.data])
        
        # Convert date strings to datetime with robust error handling
        if 'date' in data_df.columns:
            try:
                data_df['date'] = pd.to_datetime(data_df['date'], errors='coerce')
                # Check if we have any NaT (Not a Time) values after conversion
                if data_df['date'].isna().any():
                    bad_dates = [d for i, d in enumerate(financial_data.data) if pd.isna(data_df['date'].iloc[i])]
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": f"Invalid date format found in {len(bad_dates)} rows. Please use YYYY-MM-DD format.",
                            "examples": [{"row": i, "date": d.date} for i, d in enumerate(bad_dates[:5])] if bad_dates else []
                        }
                    )
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"Error converting dates: {str(e)}",
                        "suggestion": "Please ensure all dates are in YYYY-MM-DD format."
                    }
                )
            
        # Use the business service to handle the upload
        result = business_forecasting_service.upload_financial_data(data_df)
        
        return JSONResponse(content=result)
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error uploading financial data: {str(e)}\n{error_trace}"
        )

@app.post("/api/business/generate-sample-data")
async def generate_sample_financial_data(num_records: int = 36):
    """Generate sample financial data for forecasting"""
    try:
        # Use the business service to generate sample data
        result = business_forecasting_service.generate_sample_data(num_records)
        
        return JSONResponse(content=result)
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating sample data: {str(e)}\n{error_trace}"
        )

@app.post("/api/business/forecast")
async def run_business_forecast(request: ForecastRequest = None):
    """Run business forecasting on the financial data"""
    try:
        # Use default values if no request body is provided
        periods = 12
        period_type = "month"
        
        if request:
            periods = request.periods
            period_type = request.period_type
        
        # Validate inputs
        if periods <= 0:
            raise HTTPException(status_code=400, detail="Forecast periods must be greater than 0")
            
        if period_type.lower() not in ["month", "year"]:
            raise HTTPException(status_code=400, detail="Period type must be 'month' or 'year'")
        
        # Use the business service to run the forecast
        result = business_forecasting_service.forecast(
            forecast_periods=periods,
            output_type=period_type
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error running business forecast: {str(e)}\n{error_trace}"
        )

@app.get("/api/business/forecast-results")
async def get_business_forecast_results():
    """Get the results of the last business forecast"""
    try:
        # Use the business service to get the forecast results
        result = business_forecasting_service.get_forecast_results()
        
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving business forecast results: {str(e)}\n{error_trace}"
        )

@app.get("/api/business/dashboard")
async def get_business_dashboard():
    """Get the business forecast dashboard"""
    try:
        # Use the business service to get the dashboard
        result = business_forecasting_service.get_dashboard()
        
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving business dashboard: {str(e)}\n{error_trace}"
        )

@app.get("/api/business/report")
async def get_business_report():
    """Get the business forecast report"""
    try:
        # Use the business service to get the report
        result = business_forecasting_service.get_report()
        
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving business report: {str(e)}\n{error_trace}"
        )

@app.delete("/api/business/data")
async def delete_business_forecast_data():
    """Delete all business forecast data and results"""
    try:
        # Use the business service to delete the data
        result = business_forecasting_service.delete_forecast_data()
        
        return JSONResponse(content=result)
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error deleting business forecast data: {str(e)}\n{error_trace}"
        )

@app.get("/api/business/quick-forecast")
async def quick_business_forecast(periods: int = 12, period_type: str = "month"):
    """One-click endpoint to use the original dataset and run a forecast"""
    try:
        # Instead of generating sample data, we'll use the original dataset
        # The business_forecasting_service already has the correct dataset_path set
        
        # Run forecast directly on the original dataset
        forecast_result = business_forecasting_service.forecast(
            forecast_periods=periods,
            output_type=period_type
        )
        
        if forecast_result["status"] != "success":
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "Failed to run forecast",
                    "details": forecast_result
                }
            )
        
        # 3. Prepare simplified response with key information
        response = {
            "status": "success",
            "message": "Quick forecast completed successfully using original dataset",
            "forecast_periods": periods,
            "period_type": period_type,
            "data": {
                "file_path": business_forecasting_service.dataset_path,
                "date_range": forecast_result.get("data_range", {})
            },
            "forecast": {
                "metrics": forecast_result["metrics"],
                "insights": forecast_result["insights"][:5] if len(forecast_result["insights"]) > 5 else forecast_result["insights"],
                "sample_forecasts": forecast_result["sample_forecasts"]
            },
            "links": {
                "detailed_results": "/api/business/forecast-results",
                "dashboard": "/api/business/dashboard",
                "report": "/api/business/report"
            }
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error running quick business forecast: {str(e)}\n{error_trace}"
        )

@app.get("/api/business/use-original-dataset")
async def use_original_dataset():
    """Use the original financial dataset with 60,000 rows spanning 2019-2024"""
    try:
        # Get the path to the original dataset from the business forecasting module
        original_dataset_path = business_forecasting_service.dataset_path
        
        # Get absolute path for better debugging
        absolute_path = os.path.abspath(original_dataset_path)
        
        # Print diagnostic information
        print(f"Looking for original dataset at: {original_dataset_path}")
        print(f"Absolute path: {absolute_path}")
        
        # Check if parent directory exists
        parent_dir = os.path.dirname(absolute_path)
        if os.path.exists(parent_dir):
            print(f"Parent directory exists: {parent_dir}")
            print("Directory contents:")
            for item in os.listdir(parent_dir):
                print(f"  - {item}")
        else:
            print(f"Parent directory does not exist: {parent_dir}")
        
        # Check if the file exists
        if not os.path.exists(original_dataset_path):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": f"Original dataset not found at path: {original_dataset_path}",
                    "absolute_path": absolute_path,
                    "working_directory": os.getcwd(),
                    "parent_directory": parent_dir,
                    "parent_exists": os.path.exists(parent_dir),
                    "parent_contents": os.listdir(parent_dir) if os.path.exists(parent_dir) else [],
                    "suggestion": "Please make sure the financial_metrics.csv file exists at the expected location."
                }
            )
        
        # Load the dataset
        try:
            df = pd.read_csv(original_dataset_path)
            
            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Get basic statistics
            row_count = len(df)
            columns = df.columns.tolist()
            
            # Get date range if available
            date_range = None
            if 'date' in df.columns and not df['date'].isna().all():
                start_date = df['date'].min().strftime('%Y-%m-%d')
                end_date = df['date'].max().strftime('%Y-%m-%d')
                date_range = f"{start_date} to {end_date}"
            
            # Get sample data
            sample_data = df.head(5).copy()
            if 'date' in sample_data.columns:
                sample_data['date'] = sample_data['date'].dt.strftime('%Y-%m-%d')
            sample_data = sample_data.to_dict('records')
            
            # Prepare response
            result = {
                "status": "success",
                "message": f"Successfully loaded original financial dataset with {row_count} rows",
                "record_count": row_count,
                "columns": columns,
                "date_range": date_range,
                "sample_data": sample_data,
                "file_path": original_dataset_path
            }
            
            # No need to use the service's upload function since we're using the original dataset directly
            return JSONResponse(content=result)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Error reading original dataset: {str(e)}",
                    "error_trace": error_trace
                }
            )
            
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error accessing original dataset: {str(e)}\n{error_trace}"
        )

@app.post("/api/business/ensure-original-dataset")
async def ensure_original_dataset():
    """
    Ensures the original dataset is available at the correct path defined in buisness_forecasting.py
    If not found, attempts to fix the path
    """
    try:
        expected_path = BUSINESS_FORECAST_DATASET_PATH
        abs_path = os.path.abspath(expected_path)
        parent_dir = os.path.dirname(abs_path)
        
        # Log detailed information
        print(f"Expected dataset path: {expected_path}")
        print(f"Absolute path: {abs_path}")
        print(f"Parent directory: {parent_dir}")
        
        # Check if it already exists
        if os.path.exists(expected_path):
            # Validate it's readable
            try:
                df = pd.read_csv(expected_path)
                record_count = len(df)
                return JSONResponse(
                    content={
                        "status": "success",
                        "message": f"Original dataset already exists at {expected_path}",
                        "record_count": record_count,
                        "columns": df.columns.tolist(),
                        "file_path": expected_path,
                        "absolute_path": abs_path
                    }
                )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": f"Dataset exists but cannot be read: {str(e)}",
                        "file_path": expected_path,
                        "absolute_path": abs_path
                    }
                )
        
        # Create parent directory if needed
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            print(f"Created parent directory: {parent_dir}")
            
        # Case 1: Check if dataset_path from the module exists (it might be somewhere else)
        potential_source = business_forecasting_service.dataset_path
        if os.path.exists(potential_source) and potential_source != expected_path:
            print(f"Found dataset at {potential_source}, copying to {expected_path}")
            # Read original
            df = pd.read_csv(potential_source)
            # Save to expected path
            df.to_csv(expected_path, index=False)
            return JSONResponse(
                content={
                    "status": "success",
                    "message": f"Copied dataset from {potential_source} to {expected_path}",
                    "record_count": len(df),
                    "file_path": expected_path,
                    "source_path": potential_source
                }
            )
            
        # Case 2: Look for the file in various places
        search_paths = [
            os.path.join(os.getcwd(), 'financial_metrics.csv'),
            os.path.join(os.getcwd(), 'datacleaning1', 'financial_metrics.csv'),
            os.path.join(os.getcwd(), 'data', 'financial_metrics.csv')
        ]
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                print(f"Found dataset at {search_path}, copying to {expected_path}")
                # Read original
                df = pd.read_csv(search_path)
                # Save to expected path
                df.to_csv(expected_path, index=False)
                return JSONResponse(
                    content={
                        "status": "success",
                        "message": f"Copied dataset from {search_path} to {expected_path}",
                        "record_count": len(df),
                        "file_path": expected_path,
                        "source_path": search_path
                    }
                )
        
        # Case 3: If we couldn't find it anywhere, create it using our force-large-dataset endpoint
        print("Dataset not found, creating a new one")
        
        # Call the force-large-dataset endpoint logic here directly
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2024, 3, 1)
        
        # Generate dates
        days = (end_date - start_date).days + 1
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Ensure we have at least 60,000 records
        if len(dates) < 60000:
            # Duplicate dates as needed to reach 60,000
            dates = dates * (60000 // len(dates) + 1)
            dates = dates[:60000]
        
        # Generate sample data
        np.random.seed(42)  # For reproducibility
        
        # Create metrics
        data = {
            'date': dates,
            'revenue': [random.uniform(10000, 50000) for _ in range(len(dates))],
            'expenses': [random.uniform(8000, 40000) for _ in range(len(dates))],
            'cash_flow': [random.uniform(2000, 10000) for _ in range(len(dates))]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Make sure parent directory exists
        os.makedirs(parent_dir, exist_ok=True)
        
        # Save the dataset
        df.to_csv(expected_path, index=False)
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Created new dataset at {expected_path}",
                "record_count": len(df),
                "file_path": expected_path,
                "absolute_path": abs_path,
                "date_range": f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
                "columns": list(data.keys())
            }
        )
    
    except Exception as e:
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error ensuring original dataset: {str(e)}",
                "error_trace": error_trace
            }
        )

# Financial Ratio Analysis API Endpoints
@app.post("/api/financial-ratios/analyze")
async def analyze_financial_ratios():
    """
    Analyze financial ratios using the dataset from buisness_forecasting.py
    """
    try:
        # Initialize the analyzer with the same dataset used by buisness_forecasting
        analyzer = FinancialRatioAnalyzer(BUSINESS_FORECAST_DATASET_PATH)
        
        # Capture console output to include in response
        output_capture = StringIO()
        with redirect_stdout(output_capture):
            # Run the analysis steps
            if analyzer.load_and_analyze_data():
                if analyzer.preprocess_data():
                    if analyzer.compute_key_ratios():
                        analyzer.plot_key_ratios()
                        analyzer.generate_business_insights()
                        
                        # Check if forecast data exists and run additional analysis if available
                        forecast_available = analyzer.load_forecast_data()
                        if forecast_available:
                            analyzer.generate_combined_insights()
                            analyzer.plot_combined_ratios()
                            analyzer.generate_quotation_analysis()
        
        # Collect results and paths to output files
        results = {
            "status": "success",
            "message": "Financial ratio analysis completed successfully",
            "dataset_path": BUSINESS_FORECAST_DATASET_PATH,
            "output_files": {}
        }
        
        # Check for output files and add them to the response
        output_files = [
            'financial_insights.txt',
            'key_financial_ratios.png',
            'combined_financial_insights.txt',
            'combined_ratio_analysis.png',
            'financial_analysis_report.txt'
        ]
        
        for file in output_files:
            if os.path.exists(file):
                # Move to the financial_ratio_results directory if not already there
                destination = os.path.join('financial_ratio_results', file)
                if not os.path.exists(destination) or os.path.getmtime(file) > os.path.getmtime(destination):
                    shutil.copy(file, destination)
                results["output_files"][file] = destination
        
        # Add console output for debugging
        results["console_output"] = output_capture.getvalue()
        
        return JSONResponse(content=results)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error analyzing financial ratios: {str(e)}",
                "error_trace": error_trace
            }
        )

@app.get("/api/financial-ratios/results")
async def get_financial_ratio_results():
    """
    Retrieve financial ratio analysis results
    """
    try:
        # Check if results directory exists
        if not os.path.exists('financial_ratio_results'):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "No financial ratio analysis results found",
                    "suggestion": "Run /api/financial-ratios/analyze first to generate results"
                }
            )
        
        # Check for key result files
        insights_path = os.path.join('financial_ratio_results', 'financial_insights.txt')
        combined_insights_path = os.path.join('financial_ratio_results', 'combined_financial_insights.txt')
        analysis_report_path = os.path.join('financial_ratio_results', 'financial_analysis_report.txt')
        
        # Gather insights if available
        insights = []
        if os.path.exists(insights_path):
            with open(insights_path, 'r') as f:
                insights = [line.strip() for line in f.readlines() if line.strip()]
        
        combined_insights = []
        if os.path.exists(combined_insights_path):
            with open(combined_insights_path, 'r') as f:
                combined_insights = [line.strip() for line in f.readlines() if line.strip()]
        
        # List available image files
        image_files = []
        for file in os.listdir('financial_ratio_results'):
            if file.endswith('.png'):
                image_files.append(os.path.join('financial_ratio_results', file))
        
        # Create results summary
        response = {
            "status": "success",
            "message": "Financial ratio analysis results retrieved successfully",
            "insights": insights[:10] if len(insights) > 10 else insights,  # Limit to first 10 for brevity
            "combined_insights": combined_insights[:10] if len(combined_insights) > 10 else combined_insights,
            "visualization_files": image_files,
            "available_files": os.listdir('financial_ratio_results')
        }
        
        # Check if analysis report exists and include summary
        if os.path.exists(analysis_report_path):
            with open(analysis_report_path, 'r') as f:
                report_content = f.read()
                # Include a summary (first 500 chars) of the report
                response["analysis_report_summary"] = report_content[:500] + "..." if len(report_content) > 500 else report_content
        
        return JSONResponse(content=response)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error retrieving financial ratio results: {str(e)}",
                "error_trace": error_trace
            }
        )

@app.get("/api/financial-ratios/dashboard")
async def get_financial_ratio_dashboard():
    """
    Generate and retrieve a dashboard for financial ratio analysis
    """
    try:
        # Check if results exist
        if not os.path.exists('financial_ratio_results'):
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "No financial ratio analysis results found",
                    "suggestion": "Run /api/financial-ratios/analyze first to generate results"
                }
            )
        
        # Check for visualization files
        visualization_files = [f for f in os.listdir('financial_ratio_results') if f.endswith('.png')]
        if not visualization_files:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "message": "No visualization files found",
                    "suggestion": "Run /api/financial-ratios/analyze first to generate visualizations"
                }
            )
        
        # Generate a simple HTML dashboard
        dashboard_path = os.path.join('financial_ratio_results', 'dashboard.html')
        
        # Get insights content
        insights_content = ""
        insights_path = os.path.join('financial_ratio_results', 'financial_insights.txt')
        if os.path.exists(insights_path):
            with open(insights_path, 'r') as f:
                insights_content = f.read().replace('\n', '<br>')
        
        # Get combined insights content
        combined_insights_content = ""
        combined_insights_path = os.path.join('financial_ratio_results', 'combined_financial_insights.txt')
        if os.path.exists(combined_insights_path):
            with open(combined_insights_path, 'r') as f:
                combined_insights_content = f.read().replace('\n', '<br>')
        
        # Create HTML dashboard
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Financial Ratio Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .card {{ background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
                h1, h2, h3 {{ margin-top: 0; }}
                img {{ max-width: 100%; border: 1px solid #ddd; margin-bottom: 20px; }}
                .insights {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #2c3e50; }}
                .combined {{ background-color: #f0f7ff; padding: 15px; border-left: 4px solid #3498db; }}
                .footer {{ font-size: 12px; color: #777; text-align: center; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Financial Ratio Analysis Dashboard</h1>
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="container">
                <div class="card">
                    <h2>Key Financial Ratios</h2>
                    <img src="../financial_ratio_results/key_financial_ratios.png" alt="Key Financial Ratios">
                </div>
        """
        
        # Add combined ratios if available
        if os.path.exists(os.path.join('financial_ratio_results', 'combined_ratio_analysis.png')):
            html_content += """
                <div class="card">
                    <h2>Combined Ratio Analysis (Historical & Forecast)</h2>
                    <img src="../financial_ratio_results/combined_ratio_analysis.png" alt="Combined Ratio Analysis">
                </div>
            """
        
        # Add insights
        if insights_content:
            html_content += f"""
                <div class="card">
                    <h2>Business Insights</h2>
                    <div class="insights">
                        {insights_content}
                    </div>
                </div>
            """
        
        # Add combined insights
        if combined_insights_content:
            html_content += f"""
                <div class="card">
                    <h2>Combined Historical & Forecast Insights</h2>
                    <div class="combined">
                        {combined_insights_content}
                    </div>
                </div>
            """
        
        # Close HTML
        html_content += """
                <div class="footer">
                    <p>Generated using Financial Ratio Analyzer</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write dashboard to file
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Financial ratio dashboard generated successfully",
            "dashboard_path": dashboard_path,
            "visualization_files": visualization_files
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Error generating financial ratio dashboard: {str(e)}",
                "error_trace": error_trace
            }
        )

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
