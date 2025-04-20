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
# Define the correct path to the dataset (absolute path as specified by user)
DATASET_PATH = 'C:/Users/pro-tech/Desktop/acc.pro/datacleaning1/financial_metrics.csv'

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

def create_dashboard(data, forecasts, metrics, plot_paths, output_path='dashboard/index.html'):
    """
    Create an HTML dashboard with the forecasting results
    
    Parameters:
    - data: Original dataset
    - forecasts: Dictionary of forecasts for each metric
    - metrics: List of metrics that were forecasted
    - plot_paths: Paths to the forecast visualizations
    - output_path: Path to save the dashboard HTML
    
    Returns:
    - output_path: Path to the saved dashboard
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Financial Forecasting Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                color: #333;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .row {{
                display: flex;
                flex-wrap: wrap;
                margin: 0 -15px;
            }}
            .col {{
                flex: 1;
                padding: 0 15px;
                margin-bottom: 20px;
            }}
            .card {{
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
                padding: 15px;
                height: 100%;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 15px;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            .img-fluid {{
                max-width: 100%;
                height: auto;
            }}
            .metric-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
            }}
            .footer {{
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                font-size: 12px;
                color: #777;
            }}
            .insights {{
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #3498db;
                margin-bottom: 20px;
            }}
            .kpi {{
                text-align: center;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
                background-color: #ecf0f1;
            }}
            .kpi h3 {{
                margin: 0;
                font-size: 14px;
                color: #7f8c8d;
            }}
            .kpi p {{
                margin: 5px 0 0;
                font-size: 24px;
                font-weight: bold;
            }}
            .kpi.revenue {{
                background-color: #d5f5e3;
            }}
            .kpi.expenses {{
                background-color: #fadbd8;
            }}
            .kpi.cash-flow {{
                background-color: #d6eaf8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Financial Forecasting Dashboard</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="row">
    """
    
    # Add KPI values
    html_content += """
                <div class="col">
                    <div class="card">
                        <h2>Key Performance Indicators</h2>
                        <div class="row">
    """
    
    for metric in metrics:
        if metric in data.columns and metric in forecasts:
            last_value = data[metric].iloc[-1]
            forecast_values = forecasts[metric]['SARIMA']
            next_month = forecast_values[0]
            change = ((next_month - last_value) / last_value) * 100
            change_symbol = "+" if change >= 0 else ""
            
            kpi_class = metric.lower().replace('_', '-')
            
            html_content += f"""
                            <div class="col">
                                <div class="kpi {kpi_class}">
                                    <h3>{metric.replace('_', ' ').title()}</h3>
                                    <p>{last_value:,.2f}</p>
                                    <small>Next Month: {next_month:,.2f} ({change_symbol}{change:.1f}%)</small>
                                </div>
                            </div>
            """
    
    html_content += """
                        </div>
                    </div>
                </div>
            </div>
    """
    
    # Add forecast visualizations
    html_content += """
            <div class="row">
                <div class="col">
                    <div class="card">
                        <h2>Forecast Visualizations</h2>
    """
    
    for metric in metrics:
        if metric in plot_paths:
            # Use relative path for the image
            img_path = os.path.relpath(plot_paths[metric], os.path.dirname(output_path))
            html_content += f"""
                        <div class="metric-title">{metric.replace('_', ' ').title()} Forecast</div>
                        <img src="{img_path}" class="img-fluid" alt="{metric} forecast">
                        <br><br>
            """
    
    html_content += """
                    </div>
                </div>
            </div>
    """
    
    # Add insights
    # Get insights from forecast_insights.txt if it exists
    insights = []
    if os.path.exists('forecast_insights.txt'):
        with open('forecast_insights.txt', 'r') as f:
            insights = f.readlines()
    
    html_content += """
            <div class="row">
                <div class="col">
                    <div class="card">
                        <h2>Business Insights</h2>
                        <div class="insights">
    """
    
    for insight in insights:
        html_content += f"                            <p>{insight.strip()}</p>\n"
    
    html_content += """
                        </div>
                    </div>
                </div>
            </div>
    """
    
    # Add forecast tables
    html_content += """
            <div class="row">
                <div class="col">
                    <div class="card">
                        <h2>Forecast Tables</h2>
    """
    
    # Load consolidated results if available
    if os.path.exists('consolidated_results.csv'):
        try:
            forecast_df = pd.read_csv('consolidated_results.csv')
            
            # Display first 6 rows of the forecast
            html_content += """
                        <div class="table-responsive">
                            <table>
                                <thead>
                                    <tr>
            """
            
            # Add headers
            for col in forecast_df.columns:
                html_content += f"                                        <th>{col}</th>\n"
            
            html_content += """
                                    </tr>
                                </thead>
                                <tbody>
            """
            
            # Add rows (first 6 rows)
            for i, row in forecast_df.head(6).iterrows():
                html_content += "                                    <tr>\n"
                for col in forecast_df.columns:
                    cell_value = row[col]
                    # Format numeric values
                    if pd.api.types.is_numeric_dtype(forecast_df[col]):
                        html_content += f"                                        <td>{cell_value:,.2f}</td>\n"
                    else:
                        html_content += f"                                        <td>{cell_value}</td>\n"
                html_content += "                                    </tr>\n"
            
            html_content += """
                                </tbody>
                            </table>
                        </div>
            """
        except Exception as e:
            html_content += f"<p>Error loading forecast data: {str(e)}</p>"
    else:
        html_content += "<p>No forecast data available. Run the forecast first.</p>"
    
    html_content += """
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p>Financial Forecasting System | Generated using SARIMA Forecasting</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nDashboard created successfully at: {output_path}")
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
