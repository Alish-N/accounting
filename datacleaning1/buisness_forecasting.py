import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Get the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define the correct path to the dataset (one directory up from the script)
DATASET_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'financial_metrics.csv')

def analyze_dataset(file_path):
    """
    Analyze the dataset and print key information
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: Dataset file not found at {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        print("\nDataset Analysis:")
        print("-" * 50)
        print(f"Dataset path: {file_path}")
        print(f"Total number of records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print("\nFeatures in the dataset:")
        for col in df.columns:
            print(f"- {col}")
        print("\nData types:")
        print(df.dtypes)
        print("\nBasic statistics:")
        print(df.describe())
        print("\nMissing values:")
        print(df.isnull().sum())
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def preprocess_data(df):
    """
    Preprocess the data for time series analysis
    """
    try:
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Create time series objects for key metrics
        key_metrics = ['revenue', 'expenses', 'cash_flow']
        for metric in key_metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors='coerce')
        
        # Verify we have the required metrics
        missing_metrics = [metric for metric in key_metrics if metric not in df.columns]
        if missing_metrics:
            print(f"Warning: Missing required metrics: {missing_metrics}")
            return None
        
        # Aggregate daily data to monthly data
        try:
            # First, ensure we have enough data points
            if len(df) < 30:  # Minimum required for monthly aggregation
                print("Error: Insufficient data points for monthly aggregation")
                return None
            
            # Convert all columns to numeric where possible
            for col in df.columns:
                if col not in key_metrics:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Aggregate key metrics monthly (sum)
            monthly_data = df[key_metrics].resample('M').sum()
            
            # Add other metrics as monthly averages (only numeric columns)
            other_metrics = [col for col in df.columns if col not in key_metrics]
            numeric_other_metrics = df[other_metrics].select_dtypes(include=[np.number]).columns
            
            if len(numeric_other_metrics) > 0:
                monthly_data = monthly_data.join(df[numeric_other_metrics].resample('M').mean())
            
            # Remove any remaining NaN values
            monthly_data = monthly_data.fillna(method='ffill').fillna(method='bfill')
            
            # Verify we have enough monthly data points
            if len(monthly_data) < 12:  # Minimum required for forecasting
                print("Error: Insufficient monthly data points for forecasting")
                return None
            
            print("\nMonthly Data Summary:")
            print("-" * 50)
            print(f"Date range: {monthly_data.index.min()} to {monthly_data.index.max()}")
            print(f"Number of months: {len(monthly_data)}")
            print("\nMonthly statistics:")
            print(monthly_data[key_metrics].describe())
            
            # Print information about processed columns
            print("\nProcessed columns:")
            print(f"Key metrics (summed): {key_metrics}")
            print(f"Other metrics (averaged): {list(numeric_other_metrics)}")
            
            return monthly_data
            
        except Exception as e:
            print(f"Error during monthly aggregation: {str(e)}")
            print("Data types in DataFrame:")
            print(df.dtypes)
            return None
            
    except Exception as e:
        print(f"Error during data preprocessing: {str(e)}")
        return None

def perform_eda(df):
    """
    Perform Exploratory Data Analysis
    """
    # Create directory for plots if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Get numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # 1. Time Series Plots
    key_metrics = ['revenue', 'expenses', 'cash_flow']
    fig, axes = plt.subplots(len(key_metrics), 1, figsize=(15, 4*len(key_metrics)))
    fig.suptitle('Key Financial Metrics Over Time')
    
    for i, metric in enumerate(key_metrics):
        if metric in df.columns:
            df[metric].plot(ax=axes[i], title=f'{metric.replace("_", " ").title()} Over Time')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel(metric.replace("_", " ").title())
    
    plt.tight_layout()
    plt.savefig('plots/time_series_metrics.png')
    plt.close()
    
    # 2. Correlation Matrix (only for numeric columns)
    plt.figure(figsize=(15, 12))
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Financial Metrics')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # 3. Box Plots for Distribution (only for numeric columns)
    plt.figure(figsize=(15, 8))
    df[numeric_columns].boxplot()
    plt.title('Distribution of Financial Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/distribution_boxplot.png')
    plt.close()
    
    # 4. Seasonal Decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    for metric in key_metrics:
        if metric in df.columns:
            decomposition = seasonal_decompose(df[metric], period=12)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
            fig.suptitle(f'Seasonal Decomposition of {metric.replace("_", " ").title()}')
            
            decomposition.observed.plot(ax=ax1)
            ax1.set_title('Observed')
            
            decomposition.trend.plot(ax=ax2)
            ax2.set_title('Trend')
            
            decomposition.seasonal.plot(ax=ax3)
            ax3.set_title('Seasonal')
            
            decomposition.resid.plot(ax=ax4)
            ax4.set_title('Residual')
            
            plt.tight_layout()
            plt.savefig(f'plots/{metric}_decomposition.png')
            plt.close()

def train_models(data, metric, periods=36):
    """
    Train SARIMA model for forecasting
    """
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # SARIMA Model
    sarima_model = SARIMAX(train_data[metric],
                          order=(2, 1, 2),
                          seasonal_order=(1, 1, 1, 12),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
    
    sarima_results = sarima_model.fit(disp=False)
    
    # Generate forecasts
    sarima_forecast = sarima_results.get_forecast(steps=periods)
    sarima_forecast_mean = sarima_forecast.predicted_mean
    
    # Create future dates for SARIMA forecast
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
    sarima_forecast_mean.index = future_dates
    
    # Calculate model performance
    try:
        # Use the last 12 months of training data for validation
        validation_size = 12
        validation_data = train_data.iloc[-validation_size:]
        
        # Get SARIMA predictions for validation period
        sarima_pred = sarima_results.get_prediction(start=train_data.index[-validation_size],
                                                  end=train_data.index[-1])
        sarima_pred_mean = sarima_pred.predicted_mean
        
        # Calculate metrics
        metrics = {
            'SARIMA': {
                'MAE': mean_absolute_error(validation_data[metric], sarima_pred_mean),
                'RMSE': np.sqrt(mean_squared_error(validation_data[metric], sarima_pred_mean)),
                'MAPE': np.mean(np.abs((validation_data[metric] - sarima_pred_mean) / validation_data[metric])) * 100
            }
        }
        
        # Print validation period details
        print(f"\nValidation Period: {validation_data.index[0].strftime('%Y-%m')} to {validation_data.index[-1].strftime('%Y-%m')}")
        
    except Exception as e:
        print(f"Warning: Could not calculate model performance metrics: {str(e)}")
        metrics = {
            'SARIMA': {'MAE': 0, 'RMSE': 0, 'MAPE': 0}
        }
    
    # Create empty placeholder for compatibility
    placeholder_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': sarima_forecast_mean.values
    })
    
    return sarima_forecast_mean, placeholder_df, metrics

def plot_forecasts(data, metric, sarima_forecast, _, show_years=True):
    """
    Create professional visualization of historical data and forecast
    """
    # Create two visualizations: short-term and long-term
    
    # 1. Short-term visualization (next 12 months)
    plt.figure(figsize=(15, 8))
    
    # Format dates for x-axis
    months_formatter = plt.matplotlib.dates.DateFormatter('%b %Y')
    months_locator = plt.matplotlib.dates.MonthLocator(interval=3)
    
    # Create a subplot with shared x-axis
    ax = plt.subplot(111)
    ax.xaxis.set_major_formatter(months_formatter)
    ax.xaxis.set_major_locator(months_locator)
    
    # Plot historical data (last 24 months if available)
    hist_months = min(24, len(data))
    ax.plot(data.index[-hist_months:], data[metric].iloc[-hist_months:], label='Historical Data', color='blue', linewidth=2)
    
    # Plot forecast (12 months)
    forecast_months = min(12, len(sarima_forecast))
    ax.plot(sarima_forecast.index[:forecast_months], sarima_forecast.values[:forecast_months], 
           label='SARIMA Forecast', color='red', linestyle='--', linewidth=2)
    
    # Add confidence intervals for forecast
    ax.fill_between(sarima_forecast.index[:forecast_months], 
                   sarima_forecast.values[:forecast_months] * 0.9, 
                   sarima_forecast.values[:forecast_months] * 1.1,
                   color='red', alpha=0.1)
    
    # Enhance the plot
    plt.title(f'{metric.replace("_", " ").title()} - 12-Month Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{metric.replace("_", " ").title()} Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    # Rotate x labels
    plt.gcf().autofmt_xdate()
    
    # Add annotations
    last_historical = data[metric].iloc[-1]
    last_forecast_sarima = sarima_forecast.values[forecast_months-1]
    plt.annotate(f'Latest: {last_historical:,.2f}', 
                xy=(data.index[-1], last_historical),
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.annotate(f'Forecast (12m): {last_forecast_sarima:,.2f}', 
                xy=(sarima_forecast.index[forecast_months-1], last_forecast_sarima),
                xytext=(10, -30), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    plt.savefig(f'plots/{metric}_short_term_forecast.png', dpi=300)
    plt.close()
    
    # 2. Long-term visualization (all forecast periods)
    if len(sarima_forecast) > 12 and show_years:
        plt.figure(figsize=(15, 8))
        
        # Create a subplot
        ax = plt.subplot(111)
        
        # Format dates for x-axis based on forecast length
        if len(sarima_forecast) <= 24:  # Up to 2 years
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
        else:  # More than 2 years
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        
        # Plot historical data (last 3 years if available)
        hist_years = min(36, len(data))
        ax.plot(data.index[-hist_years:], data[metric].iloc[-hist_years:], label='Historical Data', color='blue', linewidth=2)
        
        # Plot full forecast
        ax.plot(sarima_forecast.index, sarima_forecast.values, 
               label='SARIMA Forecast', color='red', linestyle='--', linewidth=2)
        
        # Add confidence intervals with increasing uncertainty over time
        for i in range(0, len(sarima_forecast), 12):
            end_idx = min(i+12, len(sarima_forecast))
            confidence = 0.1 + (i/12) * 0.05  # Increasing confidence band
            ax.fill_between(sarima_forecast.index[i:end_idx], 
                           sarima_forecast.values[i:end_idx] * (1-confidence), 
                           sarima_forecast.values[i:end_idx] * (1+confidence),
                           color='red', alpha=0.1)
        
        # Enhance the plot
        years = len(sarima_forecast) // 12
        remaining_months = len(sarima_forecast) % 12
        time_desc = f"{years} Year{'s' if years != 1 else ''}"
        if remaining_months > 0:
            time_desc += f" and {remaining_months} Month{'s' if remaining_months != 1 else ''}"
            
        plt.title(f'{metric.replace("_", " ").title()} - Long-term Forecast ({time_desc})', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'{metric.replace("_", " ").title()} Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)
        
        # Rotate x labels
        plt.gcf().autofmt_xdate()
        
        # Add annotations for key forecast points
        for i in range(1, years+1):
            idx = i * 12 - 1
            if idx < len(sarima_forecast):
                year_value = sarima_forecast.values[idx]
                year_date = sarima_forecast.index[idx]
                growth = ((year_value - last_historical) / last_historical) * 100
                plt.annotate(f'Year {i}: {year_value:,.0f} ({growth:+.1f}%)', 
                            xy=(year_date, year_value),
                            xytext=(0, 10), textcoords='offset points',
                            ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'plots/{metric}_long_term_forecast.png', dpi=300)
        plt.close()
    
    return f'plots/{metric}_short_term_forecast.png'

def create_crm_output(sarima_forecast, _, metric):
    """
    Create CRM-ready forecast table
    """
    # Ensure forecast has proper date index
    forecast_df = pd.DataFrame({
        'month': sarima_forecast.index.strftime('%Y-%m'),
        f'predicted_{metric}_sarima': sarima_forecast.values.round(2)
    })
    
    # Rename for clarity
    forecast_df[f'predicted_{metric}'] = forecast_df[f'predicted_{metric}_sarima']
    
    return forecast_df

def generate_insights(df, forecasts):
    """
    Generate business insights from the data and forecasts
    """
    insights = []
    
    # Calculate historical trends using the last 12 months (consistent with dashboard)
    for metric in ['revenue', 'expenses', 'cash_flow']:
        if metric in df.columns:
            # Get last 13 months to calculate 12-month growth
            if len(df) >= 13:
                historical_growth = ((df[metric].iloc[-1] - df[metric].iloc[-13]) / df[metric].iloc[-13]) * 100
                insights.append(f"{metric.capitalize()} Analysis:")
                insights.append(f"- Historical growth (last 12 months): {historical_growth:.2f}%")
            else:
                # Use full dataset if less than 13 months available
                historical_growth = ((df[metric].iloc[-1] - df[metric].iloc[0]) / df[metric].iloc[0]) * 100
                insights.append(f"{metric.capitalize()} Analysis:")
                insights.append(f"- Historical growth (full period): {historical_growth:.2f}%")
            
            # Calculate forecast growth
            forecast_growth = ((forecasts[metric]['SARIMA'][-1] - forecasts[metric]['SARIMA'][0]) / 
                             forecasts[metric]['SARIMA'][0]) * 100
            insights.append(f"- Expected growth over next 12 months: {forecast_growth:.2f}%")
            
            # Add average monthly values
            insights.append(f"- Avg monthly {metric} (last 12m): {df[metric].iloc[-12:].mean():.2f}")
            insights.append(f"- Min monthly {metric} (last 12m): {df[metric].iloc[-12:].min():.2f}")
            insights.append(f"- Max monthly {metric} (last 12m): {df[metric].iloc[-12:].max():.2f}")
    
    # Add seasonal insights
    if 'revenue' in df.columns:
        seasonal_pattern = df['revenue'].groupby(df.index.month).mean()
        peak_month = seasonal_pattern.idxmax()
        month_names = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 
                       7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
        
        insights.append(f"\nSeasonal Insights:")
        insights.append(f"- Peak revenue typically occurs in {month_names[peak_month]}")
        
        # Add month-over-month analysis
        if len(df) >= 2:
            mom_change = ((df['revenue'].iloc[-1] - df['revenue'].iloc[-2]) / df['revenue'].iloc[-2]) * 100
            insights.append(f"- Last month-over-month change: {mom_change:.2f}%")
    
    return insights

def create_consolidated_output(forecasts_dict, metrics, insights):
    """
    Create a consolidated output with all forecasts and insights
    """
    # Create a consolidated DataFrame for all metrics
    consolidated_df = pd.DataFrame()
    
    # Add forecast data for each metric
    for metric in metrics:
        if metric in forecasts_dict:
            forecast_data = forecasts_dict[metric]
            
            # Format the month column once
            if consolidated_df.empty:
                consolidated_df['month'] = forecast_data['month']
            
            # Add the forecasts for this metric
            for column in forecast_data.columns:
                if column != 'month':
                    consolidated_df[column] = forecast_data[column]
    
    # Add insights to the consolidated data
    insights_dict = {'insights': insights}
    
    # Save consolidated data to CSV
    output_path = 'consolidated_results.csv'
    consolidated_df.to_csv(output_path, index=False)
    
    # Save insights to a separate text file
    with open('forecast_insights.txt', 'w') as f:
        for insight in insights:
            f.write(f"{insight}\n")
    
    return consolidated_df, output_path

def create_dashboard(data, forecasts, plot_paths, metrics, consolidated_df):
    """
    Create an interactive dashboard summary of forecasts
    """
    # Create dashboard directory if it doesn't exist
    if not os.path.exists('dashboard'):
        os.makedirs('dashboard')
    
    # Create an HTML dashboard
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Forecasting Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
            .card { background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }
            .row { display: flex; flex-wrap: wrap; margin: 0 -10px; }
            .col { flex: 1; padding: 0 10px; min-width: 300px; }
            h1, h2, h3 { margin-top: 0; }
            .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
            .positive { color: green; }
            .negative { color: red; }
            table { width: 100%; border-collapse: collapse; }
            th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .chart-container { height: 300px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            .footer { font-size: 12px; color: #777; text-align: center; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Financial Forecasting Dashboard</h1>
            <p>Last updated: """ + datetime.now().strftime('%Y-%m-%d') + """</p>
        </div>
        <div class="container">
    """
    
    # Add summary cards
    html_content += """
        <div class="row">
    """
    
    # Create a summary card for each metric
    for metric in metrics:
        if metric in data.columns and metric in forecasts:
            # Calculate key metrics
            current_value = data[metric].iloc[-1]
            forecast_next_month = forecasts[metric]['SARIMA'][0]
            forecast_6month = forecasts[metric]['SARIMA'][5]
            forecast_12month = forecasts[metric]['SARIMA'][11] if len(forecasts[metric]['SARIMA']) > 11 else forecasts[metric]['SARIMA'][-1]
            
            # Calculate changes
            mom_change = ((forecast_next_month - current_value) / current_value) * 100
            sixm_change = ((forecast_6month - current_value) / current_value) * 100
            twelvem_change = ((forecast_12month - current_value) / current_value) * 100
            
            # Create a card for this metric
            html_content += f"""
            <div class="col">
                <div class="card">
                    <h2>{metric.capitalize()} Summary</h2>
                    <div class="metric-value">{current_value:,.2f}</div>
                    <p>Current value (latest month)</p>
                    
                    <table>
                        <tr>
                            <th>Forecast Period</th>
                            <th>Value</th>
                            <th>Change</th>
                        </tr>
                        <tr>
                            <td>Next Month</td>
                            <td>{forecast_next_month:,.2f}</td>
                            <td class="{'positive' if mom_change >= 0 else 'negative'}">{mom_change:+.2f}%</td>
                        </tr>
                        <tr>
                            <td>6 Months</td>
                            <td>{forecast_6month:,.2f}</td>
                            <td class="{'positive' if sixm_change >= 0 else 'negative'}">{sixm_change:+.2f}%</td>
                        </tr>
                        <tr>
                            <td>12 Months</td>
                            <td>{forecast_12month:,.2f}</td>
                            <td class="{'positive' if twelvem_change >= 0 else 'negative'}">{twelvem_change:+.2f}%</td>
                        </tr>
                    </table>
                </div>
            </div>
            """
    
    html_content += """
        </div> <!-- end row -->
    """
    
    # Add forecast charts
    html_content += """
        <div class="card">
            <h2>Forecast Visualizations</h2>
            <div class="row">
    """
    
    # Add each metric chart
    for metric in metrics:
        if metric in plot_paths:
            html_content += f"""
            <div class="col">
                <h3>{metric.capitalize()} Forecast</h3>
                <div class="chart-container">
                    <img src="../{plot_paths[metric]}" alt="{metric} forecast">
                </div>
            </div>
            """
    
    html_content += """
            </div> <!-- end row -->
        </div> <!-- end card -->
    """
    
    # Add forecast table
    html_content += """
        <div class="card">
            <h2>Detailed Forecast</h2>
            <table>
                <tr>
                    <th>Month</th>
    """
    
    # Add header columns for each metric
    for metric in metrics:
        if metric in forecasts:
            html_content += f"""
                    <th>{metric.capitalize()}</th>
            """
    
    html_content += """
                </tr>
    """
    
    # Add rows of data
    months = consolidated_df['month'].tolist()
    for i, month in enumerate(months):
        html_content += f"""
                <tr>
                    <td>{month}</td>
        """
        
        # Add values for each metric
        for metric in metrics:
            if metric in forecasts:
                col_name = f'predicted_{metric}'
                if col_name in consolidated_df.columns:
                    value = consolidated_df[col_name].iloc[i]
                    html_content += f"""
                    <td>{value:,.2f}</td>
                    """
        
        html_content += """
                </tr>
        """
    
    html_content += """
            </table>
        </div> <!-- end card -->
    """
    
    # Add insights
    html_content += """
        <div class="card">
            <h2>Business Insights</h2>
            <ul>
    """
    
    # Read insights from file
    with open('forecast_insights.txt', 'r') as f:
        insights = f.read().splitlines()
    
    for insight in insights:
        if insight.strip():
            html_content += f"""
                <li>{insight}</li>
            """
    
    html_content += """
            </ul>
        </div> <!-- end card -->
    """
    
    # Close HTML document
    html_content += """
        <div class="footer">
            <p>Generated using SARIMA time series forecasting model</p>
            <p>Data range: """ + data.index[0].strftime('%Y-%m-%d') + """ to """ + data.index[-1].strftime('%Y-%m-%d') + """</p>
        </div>
    </div> <!-- end container -->
    </body>
    </html>
    """
    
    # Write HTML to file
    dashboard_path = 'dashboard/index.html'
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nInteractive dashboard created: {dashboard_path}")
    return dashboard_path

def generate_forecast_report(data, forecasts, metrics, model_metrics, output_path='forecast_report.html'):
    """
    Generate a comprehensive HTML report explaining the forecasting results
    """
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Forecasting Report (SARIMA Model)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .metric-section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .highlight {{ background-color: #e8f4f8; font-weight: bold; }}
            .good {{ color: green; }}
            .bad {{ color: red; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            .footer {{ font-size: 12px; color: #777; margin-top: 30px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Financial Forecasting Report (SARIMA Model)</h1>
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report provides a detailed analysis of historical financial metrics and forecasts for the next 12 months. 
                The forecasts were generated using SARIMA time series model with data from {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}.</p>
                
                <h3>Key Findings:</h3>
                <ul>
    """
    
    # Add key findings for each metric
    for metric in metrics:
        if metric in data.columns:
            historical_growth = ((data[metric].iloc[-1] - data[metric].iloc[-13]) / data[metric].iloc[-13]) * 100
            forecast_growth = ((forecasts[metric]['SARIMA'][5] - data[metric].iloc[-1]) / data[metric].iloc[-1]) * 100
            
            trend_desc = "increasing" if forecast_growth > 0 else "decreasing"
            growth_class = "good" if (metric == 'revenue' and forecast_growth > 0) or (metric == 'expenses' and forecast_growth < 0) else ""
            growth_class = "bad" if (metric == 'revenue' and forecast_growth < 0) or (metric == 'expenses' and forecast_growth > 0) else growth_class
            
            html_content += f"""
                    <li><strong>{metric.capitalize()}:</strong> Historical growth: <span class="{'good' if historical_growth > 0 else 'bad'}">{historical_growth:.1f}%</span> | 
                    Forecast growth: <span class="{growth_class}">{forecast_growth:.1f}%</span>. 
                    {metric.capitalize()} is {trend_desc} trend for the next 6 months.</li>
            """
    
    # Close the summary section
    html_content += """
                </ul>
            </div>
    """
    
    # Add detailed sections for each metric
    for metric in metrics:
        if metric in data.columns:
            historical_growth = ((data[metric].iloc[-1] - data[metric].iloc[-13]) / data[metric].iloc[-13]) * 100
            forecast_growth = ((forecasts[metric]['SARIMA'][-1] - forecasts[metric]['SARIMA'][0]) / forecasts[metric]['SARIMA'][0]) * 100
            
            html_content += f"""
            <div class="metric-section">
                <h2>{metric.capitalize()} Analysis</h2>
                <p>The {metric} shows a historical growth of {historical_growth:.2f}% over the last 12 months and is forecasted to 
                {"increase" if forecast_growth > 0 else "decrease"} by {abs(forecast_growth):.2f}% over the next 12 months.</p>
                
                <h3>Historical Performance</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Current {metric} (latest month)</td>
                        <td>{data[metric].iloc[-1]:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Average {metric} (last 12 months)</td>
                        <td>{data[metric].iloc[-12:].mean():,.2f}</td>
                    </tr>
                    <tr>
                        <td>Minimum {metric} (last 12 months)</td>
                        <td>{data[metric].iloc[-12:].min():,.2f}</td>
                    </tr>
                    <tr>
                        <td>Maximum {metric} (last 12 months)</td>
                        <td>{data[metric].iloc[-12:].max():,.2f}</td>
                    </tr>
                    <tr>
                        <td>Year-over-Year Growth</td>
                        <td class="{'good' if historical_growth > 0 else 'bad'}">{historical_growth:.2f}%</td>
                    </tr>
                </table>
                
                <h3>Forecast Performance</h3>
                <table>
                    <tr>
                        <th>Month</th>
                        <th>SARIMA Forecast</th>
                    </tr>
            """
            
            # Add forecast values for each month
            for i in range(len(forecasts[metric]['SARIMA'])):
                date = forecasts[metric]['SARIMA'].index[i].strftime('%b %Y')
                sarima_val = forecasts[metric]['SARIMA'][i]
                
                html_content += f"""
                    <tr>
                        <td>{date}</td>
                        <td class="highlight">{sarima_val:,.2f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h3>Model Performance</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """
            
            # Add model performance metrics
            if metric in model_metrics:
                metrics_dict = model_metrics[metric]['SARIMA']
                html_content += f"""
                <tr>
                    <td>Mean Absolute Error (MAE)</td>
                    <td>{metrics_dict['MAE']:,.2f}</td>
                </tr>
                <tr>
                    <td>Root Mean Square Error (RMSE)</td>
                    <td>{metrics_dict['RMSE']:,.2f}</td>
                </tr>
                <tr>
                    <td>Mean Absolute Percentage Error (MAPE)</td>
                    <td>{metrics_dict['MAPE']:,.2f}%</td>
                </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
    
    # Add methodology and footer
    html_content += """
            <div class="metric-section">
                <h2>Methodology</h2>
                <p>This forecast was generated using the SARIMA (Seasonal AutoRegressive Integrated Moving Average) time series model, which captures trend, seasonality, and autocorrelation patterns in the data.</p>
                <p>The model specifications used:</p>
                <ul>
                    <li>Order parameters (p,d,q): (2, 1, 2)</li>
                    <li>Seasonal order parameters (P,D,Q,s): (1, 1, 1, 12)</li>
                </ul>
                <p>The forecasts are based on historical patterns and do not account for unexpected external factors or market disruptions.</p>
            </div>
            
            <div class="footer">
                <p>Generated on """ + datetime.now().strftime('%Y-%m-%d') + """</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nComprehensive forecast report created: {output_path}")
    return output_path

def forecast_financials(dataset_path=DATASET_PATH, forecast_periods=12, output_type='month'):
    """
    Flexible forecasting function that allows custom forecast periods
    
    Parameters:
    - dataset_path: Path to the financial dataset CSV
    - forecast_periods: Number of periods to forecast (default: 12)
    - output_type: 'month' or 'year' (default: 'month')
    
    Returns:
    - Dictionary containing all forecasts, visualizations, and metrics
    """
    # Step 1: Analyze Dataset
    print(f"Loading dataset from: {dataset_path}")
    df = analyze_dataset(dataset_path)
    if df is None:
        return None
    
    # Step 2: Preprocess Data
    print("\nPreprocessing data...")
    df = preprocess_data(df)
    if df is None:
        print("Error: Data preprocessing failed. Please check your data and try again.")
        return None
    
    # Adjust forecast periods based on output type
    if output_type.lower() == 'year':
        # Convert years to months
        actual_periods = forecast_periods * 12
        print(f"\nForecasting for {forecast_periods} years ({actual_periods} months)...")
    else:
        actual_periods = forecast_periods
        print(f"\nForecasting for {forecast_periods} months...")
    
    # Step 3: Perform EDA
    perform_eda(df)
    
    # Step 4: Train Models and Generate Forecasts
    metrics = ['revenue', 'expenses', 'cash_flow']
    forecasts = {}
    crm_forecasts = {}
    plot_paths = {}
    model_performance = {}
    
    for metric in metrics:
        if metric in df.columns:
            print(f"\nProcessing {metric}...")
            
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
            
            # Print model performance
            print(f"\nModel Performance for {metric}:")
            print(f"SARIMA - MAE: {model_metrics['SARIMA']['MAE']:.2f}, RMSE: {model_metrics['SARIMA']['RMSE']:.2f}, MAPE: {model_metrics['SARIMA']['MAPE']:.2f}%")
    
    # Generate insights
    insights = generate_insights(df, forecasts)
    
    # Create consolidated output
    consolidated_df, output_path = create_consolidated_output(crm_forecasts, metrics, insights)
    
    # Create dashboard summary
    create_dashboard(df, forecasts, plot_paths, metrics, consolidated_df)
    
    # Generate comprehensive HTML report
    report_path = generate_forecast_report(df, forecasts, metrics, model_performance)
    
    print(f"\nAll forecast results saved to: {output_path}")
    print(f"Visualization plots saved to the 'plots' directory")
    print(f"Long-term forecast period: {actual_periods} months (from {df.index[-1] + pd.DateOffset(months=1):%b %Y} to {df.index[-1] + pd.DateOffset(months=actual_periods):%b %Y})")
    
    print("\nBusiness Insights:")
    for insight in insights:
        print(insight)
    
    result = {
        'forecasts': forecasts,
        'crm_forecasts': crm_forecasts,
        'plot_paths': plot_paths,
        'model_performance': model_performance,
        'insights': insights,
        'consolidated_path': output_path,
        'report_path': report_path,
        'data': df
    }
    
    return result

def main():
    """
    Main function that can be run directly or called from another script
    with custom parameters
    """
    import argparse
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Financial Metrics Forecasting Tool')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH,
                        help='Path to the financial metrics CSV file')
    parser.add_argument('--periods', type=int, default=12,
                        help='Number of periods to forecast (default: 12)')
    parser.add_argument('--type', type=str, choices=['month', 'year'], default='month',
                        help='Type of forecast period (month or year)')
    
    args = parser.parse_args()
    
    # Run forecasting with specified parameters
    forecast_financials(
        dataset_path=args.dataset,
        forecast_periods=args.periods,
        output_type=args.type
    )
    
    print("\nForecasting completed successfully!")
    print("\nTo run a custom forecast, use command-line arguments:")
    print("  --dataset: Path to the financial metrics CSV file")
    print("  --periods: Number of periods to forecast (default: 12)")
    print("  --type: Type of forecast period ('month' or 'year')")
    print("\nExample: python business_forecasting.py --periods 3 --type year")

if __name__ == "__main__":
    main()
