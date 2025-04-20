import pandas as pd
import numpy as np
from datetime import datetime
import os
from io import StringIO
from contextlib import redirect_stdout
import json
import traceback
from typing import Dict, Any, List, Optional, Union
import tempfile
import shutil

# Import the original business forecasting module
from datacleaning1.buisness_forecasting import (
    analyze_dataset, preprocess_data, perform_eda, train_models,
    plot_forecasts, create_crm_output, generate_insights,
    create_dashboard, generate_forecast_report, DATASET_PATH,
    create_consolidated_output
)

class BusinessForecastingService:
    """
    Service class for business forecasting functionality
    """
    def __init__(self):
        # Create necessary directories
        os.makedirs('forecast_results', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('dashboard', exist_ok=True)
        
        # Use the same dataset path as the original module
        self.dataset_path = self._find_actual_dataset_path()
        print(f"Using original dataset at: {self.dataset_path}")
        
        # Use the exact paths expected by the original functions
        self.output_path = 'consolidated_results.csv'
        self.insights_path = 'forecast_insights.txt'
        self.report_path = 'forecast_report.html'
        self.dashboard_path = os.path.join('dashboard', 'index.html')
        
        # Default metrics to forecast
        self.metrics = ['revenue', 'expenses', 'cash_flow']
        
        # Cache for results
        self.forecasts = {}
        self.insights = []
        self.model_performance = {}
        self.plot_paths = {}
        self.consolidated_df = None
        
        # Delete any previously created sample data files to ensure we use the original dataset
        self._delete_existing_sample_data()
        
        # Verify dataset is accessible
        self._verify_dataset_access()
        
    def _find_actual_dataset_path(self):
        """
        Find the actual path to the dataset - forcing use of the original dataset
        """
        # Use the path from the business forecasting module
        from datacleaning1.buisness_forecasting import DATASET_PATH
        original_path = DATASET_PATH
        print(f"Using dataset path from buisness_forecasting module: {original_path}")
        
        # Get absolute path for better understanding
        abs_path = os.path.abspath(original_path)
        print(f"Absolute path: {abs_path}")
        
        # If the file exists, use it
        if os.path.exists(original_path):
            try:
                # Attempt to read the file to verify it's valid
                df = pd.read_csv(original_path)
                print(f"Successfully loaded dataset with {len(df)} rows")
                
                # Get date range if available
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                    except Exception as e:
                        print(f"Error processing dates: {str(e)}")
                
                print(f"Columns: {df.columns.tolist()}")
                return original_path
            except Exception as e:
                print(f"Error reading dataset: {str(e)}")
        else:
            print(f"WARNING: Dataset not found at {original_path}")
            # Get parent directory to check what's available
            parent_dir = os.path.dirname(abs_path)
            print(f"Parent directory: {parent_dir}")
            
            if os.path.exists(parent_dir):
                print(f"Contents of parent directory:")
                for item in os.listdir(parent_dir):
                    print(f"  - {item}")
            else:
                print(f"Parent directory does not exist")
        
        # Always return the path from the module
        return original_path
        
    def _verify_dataset_access(self):
        """
        Verify that the dataset is accessible and print diagnostic information
        """
        print(f"\n===== DATASET VERIFICATION =====")
        print(f"Dataset path: {self.dataset_path}")
        
        # Check if the file exists
        if not os.path.exists(self.dataset_path):
            print(f"ERROR: Dataset file not found at: {self.dataset_path}")
            
            # Try to get more information
            abs_path = os.path.abspath(self.dataset_path)
            print(f"Absolute path: {abs_path}")
            
            # Check parent directory
            parent_dir = os.path.dirname(abs_path)
            print(f"Parent directory: {parent_dir}")
            
            if os.path.exists(parent_dir):
                print(f"Contents of parent directory:")
                for item in os.listdir(parent_dir):
                    print(f"  - {item}")
            else:
                print(f"Parent directory does not exist")
                
            # Check current directory
            cwd = os.getcwd()
            print(f"Current working directory: {cwd}")
            print(f"Contents of current directory:")
            try:
                for item in os.listdir(cwd):
                    print(f"  - {item}")
            except Exception as e:
                print(f"Error listing current directory: {str(e)}")
        else:
            print(f"Dataset file found successfully!")
            
            # Try to read the file
            try:
                df = pd.read_csv(self.dataset_path)
                print(f"Dataset loaded successfully")
                print(f"Number of rows: {len(df)}")
                print(f"Columns: {df.columns.tolist()}")
                
                # Check date range if date column exists
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                    except Exception as e:
                        print(f"Error processing dates: {str(e)}")
            except Exception as e:
                print(f"Error reading dataset: {str(e)}")
                
        print(f"================================\n")
    
    def _make_json_serializable(self, obj):
        """
        Recursively convert a dictionary or list to ensure it's JSON serializable.
        Handles pandas Timestamp objects and other non-serializable types.
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif hasattr(obj, 'isoformat') and callable(obj.isoformat):
            # Handle datetime objects
            return obj.isoformat()
        elif hasattr(obj, 'tolist') and callable(obj.tolist):
            # Handle numpy arrays
            return obj.tolist()
        elif pd.isna(obj):
            # Handle NaN/None values
            return None
        else:
            # Try to return as is, will fail if not serializable
            return obj
    
    def upload_financial_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Upload and save financial data for forecasting
        
        Args:
            data: DataFrame containing financial metrics
            
        Returns:
            dict: Status and info about the uploaded data
        """
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            
            # Verify date format
            if 'date' in data.columns:
                # Make sure date column is in datetime format
                if not pd.api.types.is_datetime64_any_dtype(data['date']):
                    try:
                        data['date'] = pd.to_datetime(data['date'], errors='coerce')
                        # Check for NaT values
                        if data['date'].isna().any():
                            # Filter out rows with invalid dates
                            invalid_date_count = data['date'].isna().sum()
                            data = data.dropna(subset=['date'])
                            if len(data) == 0:
                                return self._make_json_serializable({
                                    "status": "error",
                                    "message": f"All {invalid_date_count} dates were invalid. No valid data to save.",
                                    "suggestion": "Please provide dates in YYYY-MM-DD format."
                                })
                            else:
                                print(f"Warning: Dropped {invalid_date_count} rows with invalid dates")
                    except Exception as e:
                        return self._make_json_serializable({
                            "status": "error",
                            "message": f"Error converting dates: {str(e)}",
                            "suggestion": "Please ensure all dates are in YYYY-MM-DD format."
                        })
            
            # Save the data to the dataset path specified in the original module
            data.to_csv(self.dataset_path, index=False)
            
            # Basic analysis of the data
            columns = data.columns.tolist()
            metrics_found = [metric for metric in self.metrics if metric in columns]
            missing_metrics = [metric for metric in self.metrics if metric not in columns]
            
            # Get the date range if 'date' column exists
            date_range = None
            if 'date' in data.columns:
                try:
                    start_date = data['date'].min().strftime('%Y-%m-%d')
                    end_date = data['date'].max().strftime('%Y-%m-%d')
                    date_range = f"{start_date} to {end_date}"
                except:
                    date_range = "Invalid date format"
            
            response = {
                "status": "success",
                "message": f"Successfully uploaded financial data with {len(data)} records",
                "record_count": len(data),
                "columns": columns,
                "metrics_found": metrics_found,
                "missing_metrics": missing_metrics,
                "date_range": date_range,
                "file_path": self.dataset_path
            }
            
            # Ensure the response is JSON serializable
            return self._make_json_serializable(response)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return self._make_json_serializable({
                "status": "error",
                "message": f"Error uploading financial data: {str(e)}",
                "error_trace": error_trace
            })
    
    def generate_sample_data(self, num_records: int = 36) -> Dict[str, Any]:
        """
        Generate sample financial data for testing
        
        Args:
            num_records: Number of months of data to generate
            
        Returns:
            dict: Status and info about the generated data
        """
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
            
            # Start date for the data
            start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            start_date = start_date.replace(year=start_date.year - (num_records // 12) - 1)
            
            # Generate monthly dates
            dates = [start_date.replace(month=((start_date.month + i - 1) % 12) + 1, 
                                       year=start_date.year + ((start_date.month + i - 1) // 12)) 
                    for i in range(num_records)]
            
            # Generate sample metrics
            np.random.seed(42)  # For reproducibility
            
            # Revenue with growth trend and seasonality
            base_revenue = 10000
            growth_factor = 1.05  # 5% growth rate
            revenue = []
            for i in range(num_records):
                # Add seasonality - higher in months 11-12 (Nov-Dec), lower in 1-2 (Jan-Feb)
                month = dates[i].month
                seasonal_factor = 1.0
                if month in [11, 12]:  # Higher in Nov-Dec
                    seasonal_factor = 1.3
                elif month in [1, 2]:  # Lower in Jan-Feb
                    seasonal_factor = 0.8
                elif month in [6, 7, 8]:  # Summer
                    seasonal_factor = 1.1
                
                # Add random variation
                random_factor = np.random.normal(1, 0.05)
                
                # Calculate revenue with growth, seasonality and randomness
                value = base_revenue * (growth_factor ** (i/12)) * seasonal_factor * random_factor
                revenue.append(round(value, 2))
            
            # Expenses (somewhat correlated with revenue but more stable)
            expenses = []
            for i in range(num_records):
                # Base expense is 70% of revenue
                base_expense = revenue[i] * 0.7
                # Add random variation
                random_factor = np.random.normal(1, 0.03)  # Less variation than revenue
                expenses.append(round(base_expense * random_factor, 2))
            
            # Cash flow (revenue - expenses with some timing effects)
            cash_flow = []
            for i in range(num_records):
                # Simple calculation: revenue - expenses with some timing effects
                if i > 0:
                    # Some revenue from previous month affects cash flow
                    timing_effect = revenue[i-1] * 0.1
                else:
                    timing_effect = 0
                
                cf_value = revenue[i] - expenses[i] + timing_effect
                # Add random variation
                random_factor = np.random.normal(1, 0.08)  # More variation in cash flow
                cash_flow.append(round(cf_value * random_factor, 2))
            
            # Create DataFrame
            data = pd.DataFrame({
                'date': dates,
                'revenue': revenue,
                'expenses': expenses,
                'cash_flow': cash_flow
            })
            
            # Save the data
            data.to_csv(self.dataset_path, index=False)
            
            # Convert sample data to JSON-serializable format
            sample_data = data.head(5).to_dict(orient='records')
            
            # Convert Timestamp objects to string in sample_data
            for record in sample_data:
                if 'date' in record:
                    record['date'] = record['date'].strftime('%Y-%m-%d')
            
            response = {
                "status": "success",
                "message": f"Successfully generated {num_records} months of sample financial data",
                "record_count": len(data),
                "date_range": f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
                "metrics": self.metrics,
                "sample_data": sample_data,
                "file_path": self.dataset_path
            }
            
            # Ensure the response is JSON serializable
            return self._make_json_serializable(response)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return self._make_json_serializable({
                "status": "error",
                "message": f"Error generating sample data: {str(e)}",
                "error_trace": error_trace
            })
    
    def forecast(self, forecast_periods: int = 12, output_type: str = 'month') -> Dict[str, Any]:
        """
        Generate forecasts for the financial metrics
        
        Args:
            forecast_periods: Number of periods to forecast
            output_type: 'month' or 'year'
            
        Returns:
            dict: Forecasting results and insights
        """
        try:
            # Capture console output
            output_capture = StringIO()
            with redirect_stdout(output_capture):
                # Always use the dataset path from the buisness_forecasting module
                from datacleaning1.buisness_forecasting import DATASET_PATH
                original_dataset_path = DATASET_PATH
                print(f"Forecasting using dataset: {original_dataset_path}")
                
                # Check if it exists
                if not os.path.exists(original_dataset_path):
                    abs_path = os.path.abspath(original_dataset_path)
                    return self._make_json_serializable({
                        "status": "error",
                        "message": f"Original dataset not found at path: {original_dataset_path}",
                        "absolute_path": abs_path,
                        "suggestion": "Please make sure the financial_metrics.csv file exists in the parent directory."
                    })
                
                # Adjust forecast periods based on output type
                if output_type.lower() == 'year':
                    # Convert years to months
                    actual_periods = forecast_periods * 12
                else:
                    actual_periods = forecast_periods
                
                # Step 1: Analyze Dataset
                df = analyze_dataset(original_dataset_path)
                if df is None:
                    return self._make_json_serializable({
                        "status": "error",
                        "message": f"Failed to analyze dataset. Please check the data format at {original_dataset_path}",
                        "console_output": output_capture.getvalue()
                    })
                
                # Step 2: Preprocess Data
                df = preprocess_data(df)
                if df is None:
                    return self._make_json_serializable({
                        "status": "error",
                        "message": "Data preprocessing failed. Please check your data and try again.",
                        "console_output": output_capture.getvalue()
                    })
                
                # Step 3: Perform EDA
                perform_eda(df)
                
                # Step 4: Train Models and Generate Forecasts
                forecasts = {}
                crm_forecasts = {}
                plot_paths = {}
                model_performance = {}
                
                for metric in self.metrics:
                    if metric in df.columns:
                        # Train models and generate forecasts
                        sarima_forecast, placeholder, model_metrics = train_models(df, metric, periods=actual_periods)
                        forecasts[metric] = {
                            'SARIMA': sarima_forecast
                        }
                        
                        # Store model performance metrics
                        model_performance[metric] = model_metrics
                        
                        # Create visualization
                        plot_path = plot_forecasts(df, metric, sarima_forecast, None)
                        plot_paths[metric] = plot_path
                        
                        # Create CRM output
                        forecast_table = create_crm_output(sarima_forecast, None, metric)
                        crm_forecasts[metric] = forecast_table
                
                # Generate insights
                insights = generate_insights(df, forecasts)
                
                # Create consolidated output using the original function
                consolidated_df, output_path = create_consolidated_output(crm_forecasts, self.metrics, insights)
                
                # Create dashboard summary
                dashboard_path = create_dashboard(df, forecasts, self.metrics, plot_paths, 'dashboard/index.html')
                
                # Generate comprehensive HTML report
                report_path = generate_forecast_report(df, forecasts, self.metrics, model_performance)
                
                # Store results in the service
                self.forecasts = forecasts
                self.insights = insights
                self.model_performance = model_performance
                self.plot_paths = plot_paths
                self.consolidated_df = consolidated_df
                
                # Prepare response
                response = {
                    "status": "success",
                    "message": f"Financial forecasting completed successfully for {actual_periods} periods",
                    "forecast_periods": actual_periods,
                    "output_type": output_type,
                    "metrics": list(forecasts.keys()),
                    "data_range": {
                        "historical_start": df.index[0].strftime('%Y-%m-%d'),
                        "historical_end": df.index[-1].strftime('%Y-%m-%d'),
                        "forecast_start": df.index[-1] + pd.DateOffset(months=1),
                        "forecast_end": df.index[-1] + pd.DateOffset(months=actual_periods)
                    },
                    "output_files": {
                        "consolidated_results": output_path,
                        "insights": self.insights_path,
                        "dashboard": self.dashboard_path,
                        "report": self.report_path,
                        "plots": {metric: path for metric, path in plot_paths.items()}
                    },
                    "insights": insights,
                    "console_output": output_capture.getvalue()
                }
                
                # Add sample forecast values
                sample_forecasts = {}
                for metric in forecasts:
                    metric_forecasts = forecasts[metric]['SARIMA']
                    # Take first, middle, and last points for a sample
                    if len(metric_forecasts) > 0:
                        sample_forecasts[metric] = {
                            "first_period": {
                                "date": metric_forecasts.index[0].strftime('%Y-%m-%d'),
                                "value": float(metric_forecasts.iloc[0])
                            },
                            "last_period": {
                                "date": metric_forecasts.index[-1].strftime('%Y-%m-%d'),
                                "value": float(metric_forecasts.iloc[-1])
                            }
                        }
                
                response["sample_forecasts"] = sample_forecasts
                
                # Ensure the response is JSON serializable
                return self._make_json_serializable(response)
                
        except Exception as e:
            error_trace = traceback.format_exc()
            return self._make_json_serializable({
                "status": "error",
                "message": f"Error during forecasting: {str(e)}",
                "error_trace": error_trace,
                "console_output": output_capture.getvalue() if 'output_capture' in locals() else None
            })
    
    def get_forecast_results(self) -> Dict[str, Any]:
        """
        Get the results of the last forecast run
        
        Returns:
            dict: Forecast results or error message
        """
        try:
            # Check if consolidated results file exists
            consolidated_path = 'consolidated_results.csv'
            insights_path = 'forecast_insights.txt'
            
            if not os.path.exists(consolidated_path):
                return self._make_json_serializable({
                    "status": "error",
                    "message": "No forecast results found. Please run a forecast first.",
                    "suggestion": "Call the forecast method or the API endpoint to generate forecasts."
                })
            
            # Load consolidated results
            consolidated_df = pd.read_csv(consolidated_path)
            
            # Check if insights file exists
            insights = []
            if os.path.exists(insights_path):
                with open(insights_path, 'r') as f:
                    insights = [line.strip() for line in f.readlines() if line.strip()]
            
            # Check available metrics from the consolidated file
            available_metrics = []
            for metric in self.metrics:
                col_name = f'predicted_{metric}'
                if col_name in consolidated_df.columns:
                    available_metrics.append(metric)
            
            # Structure the forecast data by time period
            forecast_by_period = []
            for _, row in consolidated_df.iterrows():
                period_data = {"month": row['month']}
                for metric in available_metrics:
                    col_name = f'predicted_{metric}'
                    if col_name in row:
                        period_data[metric] = float(row[col_name])
                forecast_by_period.append(period_data)
            
            # Prepare the response
            response = {
                "status": "success",
                "message": "Forecast results retrieved successfully",
                "metrics": available_metrics,
                "insights": insights,
                "forecast_by_period": forecast_by_period
            }
            
            # Add links to output files if they exist
            output_files = {}
            if os.path.exists(consolidated_path):
                output_files["consolidated_results"] = consolidated_path
            if os.path.exists(insights_path):
                output_files["insights"] = insights_path
            if os.path.exists(self.dashboard_path):
                output_files["dashboard"] = self.dashboard_path
            if os.path.exists(self.report_path):
                output_files["report"] = self.report_path
            
            if output_files:
                response["output_files"] = output_files
            
            # Ensure the response is JSON serializable
            return self._make_json_serializable(response)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return self._make_json_serializable({
                "status": "error",
                "message": f"Error retrieving forecast results: {str(e)}",
                "error_trace": error_trace
            })
    
    def get_dashboard(self) -> Dict[str, Any]:
        """
        Get the dashboard HTML content
        
        Returns:
            dict: Dashboard HTML or error message
        """
        try:
            # Check if dashboard file exists
            dashboard_path = os.path.join('dashboard', 'index.html')
            
            if not os.path.exists(dashboard_path):
                return self._make_json_serializable({
                    "status": "error",
                    "message": "No dashboard found. Please run a forecast first.",
                    "suggestion": "Call the forecast method or the API endpoint to generate forecasts."
                })
            
            # Read dashboard HTML
            with open(dashboard_path, 'r') as f:
                dashboard_html = f.read()
            
            response = {
                "status": "success",
                "message": "Dashboard retrieved successfully",
                "dashboard_html": dashboard_html,
                "dashboard_path": dashboard_path
            }
            
            # Ensure the response is JSON serializable
            return self._make_json_serializable(response)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return self._make_json_serializable({
                "status": "error",
                "message": f"Error retrieving dashboard: {str(e)}",
                "error_trace": error_trace
            })
    
    def get_report(self) -> Dict[str, Any]:
        """
        Get the forecast report HTML content
        
        Returns:
            dict: Report HTML or error message
        """
        try:
            # Check if report file exists
            report_path = 'forecast_report.html'
            
            if not os.path.exists(report_path):
                return self._make_json_serializable({
                    "status": "error",
                    "message": "No forecast report found. Please run a forecast first.",
                    "suggestion": "Call the forecast method or the API endpoint to generate forecasts."
                })
            
            # Read report HTML
            with open(report_path, 'r') as f:
                report_html = f.read()
            
            response = {
                "status": "success",
                "message": "Forecast report retrieved successfully",
                "report_html": report_html,
                "report_path": report_path
            }
            
            # Ensure the response is JSON serializable
            return self._make_json_serializable(response)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return self._make_json_serializable({
                "status": "error",
                "message": f"Error retrieving forecast report: {str(e)}",
                "error_trace": error_trace
            })
    
    def delete_forecast_data(self) -> Dict[str, Any]:
        """
        Delete all forecast data and results
        
        Returns:
            dict: Status of the operation
        """
        try:
            files_to_delete = [
                'consolidated_results.csv',
                'forecast_insights.txt',
                'forecast_report.html'
            ]
            
            deleted_files = []
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_files.append(file_path)
            
            # Delete dashboard directory
            dashboard_path = os.path.join('dashboard', 'index.html')
            if os.path.exists(dashboard_path):
                os.remove(dashboard_path)
                deleted_files.append(dashboard_path)
            
            # Delete all plots
            plot_files = []
            for root, dirs, files in os.walk('plots'):
                for file in files:
                    if file.endswith('.png'):
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        plot_files.append(file_path)
            
            # Reset cached results
            self.forecasts = {}
            self.insights = []
            self.model_performance = {}
            self.plot_paths = {}
            self.consolidated_df = None
            
            return self._make_json_serializable({
                "status": "success",
                "message": "All forecast data and results deleted successfully",
                "deleted_files": deleted_files,
                "deleted_plots": len(plot_files)
            })
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return self._make_json_serializable({
                "status": "error",
                "message": f"Error deleting forecast data: {str(e)}",
                "error_trace": error_trace
            })

    def _delete_existing_sample_data(self):
        """
        Delete any previously created sample data files to ensure we use the original dataset
        """
        try:
            # Get path from buisness_forecasting module
            from datacleaning1.buisness_forecasting import DATASET_PATH
            original_dataset = DATASET_PATH
            
            # Check if the original dataset exists
            if not os.path.exists(original_dataset):
                print(f"WARNING: Dataset not found at expected location: {original_dataset}")
                print(f"Absolute path: {os.path.abspath(original_dataset)}")
                print("Please ensure this file exists before running forecasts")
            else:
                print(f"Dataset found at: {original_dataset}")
                print(f"Will use this for forecasting")
            
            # List of common paths where sample data might be stored (NOT INCLUDING ORIGINAL DATASET)
            sample_data_files = [
                'financial_metrics.csv',  # Local sample data
                'sample_data.csv',
                'forecast_results/financial_metrics.csv',
                'forecast_results/sample_data.csv',
                'forecast_insights.txt',  # Delete previous results to force regeneration
                'consolidated_results.csv',
                'forecast_report.html'
            ]
            
            # Make very sure we're not deleting the original dataset
            safe_paths = [path for path in sample_data_files if os.path.abspath(path) != os.path.abspath(original_dataset)]
            
            deleted_files = []
            for file_path in safe_paths:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        print(f"Deleted sample data file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {str(e)}")
            
            # Delete all plot files to force regeneration
            for root, dirs, files in os.walk('plots'):
                for file in files:
                    if file.endswith('.png'):
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            deleted_files.append(file_path)
                        except Exception as e:
                            print(f"Error deleting plot {file_path}: {str(e)}")
            
            # Clear the dashboard directory
            dashboard_path = os.path.join('dashboard', 'index.html')
            if os.path.exists(dashboard_path):
                try:
                    os.remove(dashboard_path)
                    deleted_files.append(dashboard_path)
                except Exception as e:
                    print(f"Error deleting dashboard {dashboard_path}: {str(e)}")
                    
            print(f"Cleaned up {len(deleted_files)} sample data files")
            print(f"Will use original dataset at: {original_dataset}")
                
        except Exception as e:
            print(f"Error during sample data cleanup: {str(e)}")
            traceback.print_exc() 