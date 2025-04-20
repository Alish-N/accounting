import requests
import json

print("Testing business forecasting API with updated dataset path...")

# First, let's use the original dataset endpoint
try:
    print("\n1. Using original dataset endpoint...")
    response = requests.get("http://localhost:8000/api/business/use-original-dataset")
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text[:200]}..." if len(response.text) > 200 else response.text)
except Exception as e:
    print(f"Error calling use-original-dataset: {str(e)}")

# Then, run the forecast endpoint
try:
    print("\n2. Running business forecast...")
    response = requests.post("http://localhost:8000/api/business/forecast", 
                           json={"periods": 12, "period_type": "month"})
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("API call successful!")
        print(f"Message: {data.get('message', 'No message')}")
        
        # Print forecast details if available
        if 'forecast_data' in data:
            print("\nForecast data summary:")
            forecast = data['forecast_data']
            metrics = forecast.get('metrics', [])
            print(f"- Number of metrics: {len(metrics)}")
            print(f"- Metrics forecasted: {', '.join(metrics)}")
            
            periods = forecast.get('forecast_periods', 0)
            print(f"- Forecast periods: {periods}")
            
            if 'date_range' in forecast:
                print(f"- Date range: {forecast['date_range'].get('start')} to {forecast['date_range'].get('end')}")
    else:
        print(f"Response: {response.text[:200]}..." if len(response.text) > 200 else response.text)
except Exception as e:
    print(f"Error calling business/forecast: {str(e)}")

# Get forecast results
try:
    print("\n3. Getting forecast results...")
    response = requests.get("http://localhost:8000/api/business/forecast-results")
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("API call successful!")
        print(f"Status: {data.get('status', 'No status')}")
        print(f"Message: {data.get('message', 'No message')}")
        
        # Print model performance if available
        if 'model_performance' in data:
            print("\nModel performance:")
            for model, metrics in data['model_performance'].items():
                print(f"- {model}: MAE={metrics.get('MAE', 'N/A')}, RMSE={metrics.get('RMSE', 'N/A')}")
                
        # Print sample forecast points
        if 'forecast_data' in data and 'data_points' in data['forecast_data']:
            points = data['forecast_data']['data_points']
            print(f"\nSample forecast points ({len(points)} total):")
            for i, point in enumerate(points[:3]):
                print(f"{i+1}. Date: {point.get('date')}")
                for metric, value in point.items():
                    if metric != 'date':
                        print(f"   {metric}: {value}")
                if i < 2:
                    print()
    else:
        print(f"Response: {response.text[:200]}..." if len(response.text) > 200 else response.text)
except Exception as e:
    print(f"Error calling business/forecast-results: {str(e)}")

print("\nAPI testing completed.") 