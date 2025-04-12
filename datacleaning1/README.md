# Business Forecasting System

This system provides financial forecasting capabilities using ARIMA and Facebook Prophet models.

## Features
- Data preprocessing and cleaning
- Time-series forecasting using ARIMA and Prophet
- 3-6 months and yearly predictions
- Interactive dashboard with Plotly
- Revenue, expense, and profit forecasting
- Growth trend analysis

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Ensure your financial_metrics.csv file is in the same directory
   - Required columns: date, revenue, expenses, profit_margin, etc.

3. Run the forecasting system:
```bash
python business_forecasting.py
```

## Data Preprocessing
- Handles missing values
- Removes outliers using IQR method
- Calculates growth rates and additional metrics
- Sorts and validates time-series data

## Forecasting Models
- ARIMA (Auto-Regressive Integrated Moving Average)
- Facebook Prophet (handles seasonality and holidays)

## Dashboard
The interactive dashboard shows:
- Revenue forecasts
- Expense predictions
- Profit projections
- Growth trends
- Monthly metrics
- Model comparison

## Output
The system generates:
- Short-term forecasts (3-6 months)
- Long-term predictions (yearly)
- Interactive visualizations
- Growth trend analysis 