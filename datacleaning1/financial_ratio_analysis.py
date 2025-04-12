import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import argparse
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import json

class DataUpdateHandler(FileSystemEventHandler):
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.last_updated = datetime.now()
        self.update_cooldown = 5  # seconds

    def on_modified(self, event):
        if event.src_path == self.analyzer.dataset_path:
            current_time = datetime.now()
            if (current_time - self.last_updated).total_seconds() > self.update_cooldown:
                print(f"\nDetected changes in {event.src_path}")
                self.analyzer.update_data()
                self.last_updated = current_time

class FinancialRatioAnalyzer:
    def __init__(self, dataset_path=None, auto_update=False):
        """
        Initialize the Financial Ratio Analyzer
        
        Parameters:
        dataset_path (str): Path to the CSV file containing financial metrics
        auto_update (bool): Whether to enable automatic updates
        """
        self.dataset_path = dataset_path
        self.df = None
        self.ratios = {}
        self.last_update = None
        self.auto_update = auto_update
        self.observer = None
        self.update_thread = None
        self.update_interval = 300  # 5 minutes
        
        # Create plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Create cache directory for storing analysis results
        if not os.path.exists('cache'):
            os.makedirs('cache')
    
    def start_auto_update(self):
        """Start automatic data update monitoring"""
        if self.auto_update and self.dataset_path:
            # File system monitoring
            self.observer = Observer()
            event_handler = DataUpdateHandler(self)
            self.observer.schedule(event_handler, path=os.path.dirname(self.dataset_path), recursive=False)
            self.observer.start()
            
            # Periodic checking thread
            self.update_thread = threading.Thread(target=self._periodic_update, daemon=True)
            self.update_thread.start()
            
            print(f"Auto-update enabled. Monitoring {self.dataset_path} for changes...")
    
    def stop_auto_update(self):
        """Stop automatic data update monitoring"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        self.auto_update = False
        print("Auto-update disabled.")
    
    def _periodic_update(self):
        """Periodically check for data updates"""
        while self.auto_update:
            time.sleep(self.update_interval)
            self.update_data()
    
    def update_data(self):
        """Update data and recalculate ratios"""
        try:
            # Load new data
            new_df = pd.read_csv(self.dataset_path)
            new_df['date'] = pd.to_datetime(new_df['date'])
            
            # Check if there are actually new records
            if self.df is not None:
                if len(new_df) > len(self.df):
                    print("\nNew data detected!")
                    self.df = new_df
                    self.last_update = datetime.now()
                    
                    # Recalculate ratios
                    self.calculate_all_ratios()
                    
                    # Update visualizations
                    self.plot_ratio_trends()
                    
                    # Cache the results
                    self._cache_results()
                    
                    print(f"Analysis updated at {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Display new insights
                    print("\nLatest Financial Insights:")
                    print("-" * 50)
                    self.display_key_insights()
                else:
                    print("No new data found.")
            else:
                self.df = new_df
                self.last_update = datetime.now()
                self.calculate_all_ratios()
                
        except Exception as e:
            print(f"Error updating data: {str(e)}")
    
    def _cache_results(self):
        """Cache analysis results"""
        cache_data = {
            'last_update': self.last_update.isoformat(),
            'ratios': {name: values.tolist() for name, values in self.ratios.items()},
            'latest_insights': self._get_latest_insights()
        }
        
        try:
            with open('cache/analysis_cache.json', 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error caching results: {str(e)}")
    
    def _get_latest_insights(self):
        """Get latest insights for caching"""
        insights = {}
        
        if 'current_ratio' in self.ratios:
            insights['current_ratio'] = {
                'value': float(self.ratios['current_ratio'].iloc[-1]),
                'interpretation': self.get_ratio_interpretation('current_ratio', self.ratios['current_ratio'].iloc[-1])
            }
        
        if 'net_profit_margin' in self.ratios:
            insights['net_profit_margin'] = {
                'value': float(self.ratios['net_profit_margin'].iloc[-1]),
                'interpretation': self.get_ratio_interpretation('net_profit_margin', self.ratios['net_profit_margin'].iloc[-1])
            }
        
        return insights

    def load_data(self):
        """
        Load and validate the financial dataset
        """
        try:
            if self.dataset_path is None:
                # Create sample data from 2019 to 2024
                dates = pd.date_range(start='2019-01-01', end='2024-12-31', freq='ME')
                self.df = pd.DataFrame({
                    'date': dates,
                    'current_assets': np.random.uniform(800000, 1200000, len(dates)),
                    'current_liabilities': np.random.uniform(400000, 600000, len(dates)),
                    'total_assets': np.random.uniform(2000000, 2500000, len(dates)),
                    'total_liabilities': np.random.uniform(1000000, 1500000, len(dates)),
                    'total_equity': np.random.uniform(800000, 1200000, len(dates)),
                    'revenue': np.random.uniform(300000, 500000, len(dates)),
                    'net_income': np.random.uniform(50000, 100000, len(dates)),
                    'operating_income': np.random.uniform(80000, 150000, len(dates)),
                    'inventory': np.random.uniform(200000, 300000, len(dates)),
                    'accounts_receivable': np.random.uniform(150000, 250000, len(dates)),
                    'accounts_payable': np.random.uniform(100000, 200000, len(dates)),
                    'cash_flow': np.random.uniform(40000, 80000, len(dates))
                })
                print("Using sample data (2019-2024) for demonstration...")
            else:
                self.df = pd.read_csv(self.dataset_path)
            
            # Convert date column to datetime
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df.set_index('date', inplace=True)
            
            # Start auto-update if enabled
            if self.auto_update:
                self.start_auto_update()
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return False
    
    def get_ratio_interpretation(self, ratio_name, value):
        """Get interpretation for a given ratio value"""
        interpretations = {
            'current_ratio': {
                'ranges': [(2.0, float('inf'), "Excellent liquidity position"),
                          (1.5, 2.0, "Good liquidity (can cover short-term debts)"),
                          (1.0, 1.5, "Adequate liquidity but could improve"),
                          (0, 1.0, "Potential liquidity issues")]
            },
            'quick_ratio': {
                'ranges': [(1.5, float('inf'), "Strong immediate liquidity"),
                          (1.0, 1.5, "Good immediate liquidity"),
                          (0.7, 1.0, "Adequate quick ratio"),
                          (0, 0.7, "May have difficulty meeting immediate obligations")]
            },
            'debt_to_equity': {
                'ranges': [(0, 1.0, "Conservative financial structure"),
                          (1.0, 2.0, "Moderate leverage"),
                          (2.0, float('inf'), "High leverage (high financial risk)")]
            },
            'net_profit_margin': {
                'ranges': [(20, float('inf'), "Excellent profitability"),
                          (10, 20, "Good profitability"),
                          (5, 10, "Moderate profitability"),
                          (0, 5, "Low profitability")]
            },
            'roa': {
                'ranges': [(10, float('inf'), "Excellent asset utilization"),
                          (5, 10, "Good asset utilization"),
                          (3, 5, "Moderate asset utilization"),
                          (0, 3, "Poor asset utilization")]
            },
            'roe': {
                'ranges': [(20, float('inf'), "Excellent return to shareholders"),
                          (15, 20, "Good return to shareholders"),
                          (10, 15, "Moderate return to shareholders"),
                          (0, 10, "Poor return to shareholders")]
            },
            'asset_turnover': {
                'ranges': [(2.0, float('inf'), "Excellent asset efficiency"),
                          (1.0, 2.0, "Good asset efficiency"),
                          (0.5, 1.0, "Moderate asset efficiency"),
                          (0, 0.5, "Poor asset efficiency")]
            }
        }
        
        if ratio_name not in interpretations:
            return "No interpretation available"
        
        for low, high, interpretation in interpretations[ratio_name]['ranges']:
            if low <= value < high:
                return interpretation
        
        return "No interpretation available"

    def calculate_and_display_ratios(self):
        """Calculate and display all financial ratios in a structured format"""
        # Calculate all ratios first
        self.calculate_all_ratios()
        
        print("\nFINANCIAL RATIO ANALYSIS")
        print("-" * 75)
        print(f"{'Ratio Name':<25} {'Value':>10} {'Interpretation':<35}")
        print("-" * 75)
        
        # Display Liquidity Ratios
        if 'current_ratio' in self.ratios:
            value = self.ratios['current_ratio'].iloc[-1]
            print(f"Current Ratio{':':.<14} {value:>10.2f} {self.get_ratio_interpretation('current_ratio', value)}")
        
        if 'quick_ratio' in self.ratios:
            value = self.ratios['quick_ratio'].iloc[-1]
            print(f"Quick Ratio{':':.<15} {value:>10.2f} {self.get_ratio_interpretation('quick_ratio', value)}")
        
        print("")  # Add spacing between sections
        
        # Display Profitability Ratios
        if 'net_profit_margin' in self.ratios:
            value = self.ratios['net_profit_margin'].iloc[-1]
            print(f"Net Profit Margin{':':.<10} {value:>10.2f}% {self.get_ratio_interpretation('net_profit_margin', value)}")
        
        if 'roa' in self.ratios:
            value = self.ratios['roa'].iloc[-1]
            print(f"Return on Assets{':':.<11} {value:>10.2f}% {self.get_ratio_interpretation('roa', value)}")
        
        if 'roe' in self.ratios:
            value = self.ratios['roe'].iloc[-1]
            print(f"Return on Equity{':':.<11} {value:>10.2f}% {self.get_ratio_interpretation('roe', value)}")
        
        print("")  # Add spacing between sections
        
        # Display Efficiency Ratios
        if 'asset_turnover' in self.ratios:
            value = self.ratios['asset_turnover'].iloc[-1]
            print(f"Asset Turnover{':':.<13} {value:>10.2f} {self.get_ratio_interpretation('asset_turnover', value)}")
        
        if 'inventory_turnover' in self.ratios:
            value = self.ratios['inventory_turnover'].iloc[-1]
            print(f"Inventory Turnover{':':.<10} {value:>10.2f} Times per year")
        
        print("")  # Add spacing between sections
        
        # Display Leverage Ratios
        if 'debt_to_equity' in self.ratios:
            value = self.ratios['debt_to_equity'].iloc[-1]
            print(f"Debt-to-Equity{':':.<13} {value:>10.2f} {self.get_ratio_interpretation('debt_to_equity', value)}")
        
        if 'debt_ratio' in self.ratios:
            value = self.ratios['debt_ratio'].iloc[-1]
            print(f"Debt Ratio{':':.<16} {value:>10.2f} Portion of assets financed by debt")
        
        print("\nKey Insights:")
        print("-" * 75)
        self.display_key_insights()

    def calculate_all_ratios(self):
        """Calculate all financial ratios"""
        if all(col in self.df.columns for col in ['current_assets', 'current_liabilities']):
            self.ratios['current_ratio'] = self.df['current_assets'] / self.df['current_liabilities']
            
            if 'inventory' in self.df.columns:
                quick_assets = self.df['current_assets'] - self.df['inventory']
                self.ratios['quick_ratio'] = quick_assets / self.df['current_liabilities']
        
        if all(col in self.df.columns for col in ['revenue', 'net_income']):
            self.ratios['net_profit_margin'] = (self.df['net_income'] / self.df['revenue']) * 100
            
            if 'total_assets' in self.df.columns:
                self.ratios['roa'] = (self.df['net_income'] / self.df['total_assets']) * 100
            
            if 'total_equity' in self.df.columns:
                self.ratios['roe'] = (self.df['net_income'] / self.df['total_equity']) * 100
        
        if all(col in self.df.columns for col in ['revenue', 'total_assets']):
            self.ratios['asset_turnover'] = self.df['revenue'] / self.df['total_assets']
            
            if 'inventory' in self.df.columns:
                self.ratios['inventory_turnover'] = self.df['revenue'] / self.df['inventory']
        
        if all(col in self.df.columns for col in ['total_liabilities', 'total_equity']):
            self.ratios['debt_to_equity'] = self.df['total_liabilities'] / self.df['total_equity']
            
            if 'total_assets' in self.df.columns:
                self.ratios['debt_ratio'] = self.df['total_liabilities'] / self.df['total_assets']

    def display_key_insights(self):
        """Display key insights based on the calculated ratios"""
        if 'current_ratio' in self.ratios:
            current_ratio = self.ratios['current_ratio'].iloc[-1]
            if current_ratio < 1.0:
                print("- Liquidity: Potential difficulty meeting short-term obligations")
            elif current_ratio > 2.0:
                print("- Liquidity: Strong position to cover short-term debts")
        
        if 'net_profit_margin' in self.ratios:
            npm = self.ratios['net_profit_margin'].iloc[-1]
            if npm > 20:
                print("- Profitability: Excellent operational efficiency")
            elif npm < 5:
                print("- Profitability: Need for cost control and pricing strategy review")
        
        if 'debt_to_equity' in self.ratios:
            dte = self.ratios['debt_to_equity'].iloc[-1]
            if dte > 2.0:
                print("- Leverage: High financial risk due to significant debt")
            elif dte < 1.0:
                print("- Leverage: Conservative financial structure with low risk")

    def plot_ratio_trends(self):
        """
        Create visualizations for ratio trends
        """
        if not self.ratios:
            print("No ratios calculated yet. Please calculate ratios first.")
            return False
        
        # Group ratios by category
        ratio_categories = {
            'Liquidity Ratios': ['current_ratio', 'quick_ratio'],
            'Profitability Ratios': ['net_profit_margin', 'roa', 'roe'],
            'Efficiency Ratios': ['asset_turnover', 'inventory_turnover'],
            'Leverage Ratios': ['debt_to_equity', 'debt_ratio']
        }
        
        for category, ratios in ratio_categories.items():
            # Filter available ratios
            available_ratios = [r for r in ratios if r in self.ratios]
            
            if available_ratios:
                plt.figure(figsize=(15, 6))
                
                for ratio in available_ratios:
                    plt.plot(self.ratios[ratio].index, self.ratios[ratio].values, label=ratio.replace('_', ' ').title())
                
                plt.title(f'{category} Over Time (2019-2024)')
                plt.xlabel('Date')
                plt.ylabel('Ratio Value')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save plot
                filename = f"plots/{category.lower().replace(' ', '_')}.png"
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                plt.close()
        
        return True

    def analyze_with_forecasts(self, forecast_data=None):
        """Analyze financial ratios with forecasting insights"""
        # First calculate current ratios
        self.calculate_all_ratios()
        
        print("\nCURRENT FINANCIAL HEALTH AND FUTURE OUTLOOK")
        print("=" * 75)
        
        # Current Financial Health
        print("\n1. CURRENT FINANCIAL POSITION:")
        print("-" * 75)
        
        if 'current_ratio' in self.ratios:
            current_ratio = self.ratios['current_ratio'].iloc[-1]
            print(f"Liquidity (Current Ratio): {current_ratio:.2f}")
            print(f"Status: {self.get_ratio_interpretation('current_ratio', current_ratio)}")
        
        if 'debt_to_equity' in self.ratios:
            dte = self.ratios['debt_to_equity'].iloc[-1]
            print(f"\nLeverage (Debt-to-Equity): {dte:.2f}")
            print(f"Status: {self.get_ratio_interpretation('debt_to_equity', dte)}")
        
        if 'net_profit_margin' in self.ratios:
            npm = self.ratios['net_profit_margin'].iloc[-1]
            print(f"\nProfitability (Net Profit Margin): {npm:.2f}%")
            print(f"Status: {self.get_ratio_interpretation('net_profit_margin', npm)}")
        
        # Future Outlook (if forecast data is provided)
        if forecast_data is not None:
            print("\n2. FUTURE OUTLOOK (Based on Forecasts):")
            print("-" * 75)
            
            # Get forecasted values
            if 'revenue' in forecast_data:
                next_month = forecast_data['revenue']['SARIMA'][0]
                six_month = forecast_data['revenue']['SARIMA'][5]
                twelve_month = forecast_data['revenue']['SARIMA'][11]
                
                current_revenue = self.df['revenue'].iloc[-1]
                
                print("\nRevenue Forecast:")
                print(f"Current: ${current_revenue:,.2f}")
                print(f"Next Month: ${next_month:,.2f} ({((next_month/current_revenue)-1)*100:+.1f}%)")
                print(f"6 Months: ${six_month:,.2f} ({((six_month/current_revenue)-1)*100:+.1f}%)")
                print(f"12 Months: ${twelve_month:,.2f} ({((twelve_month/current_revenue)-1)*100:+.1f}%)")
            
            if 'net_income' in self.df.columns:
                current_margin = (self.df['net_income'].iloc[-1] / self.df['revenue'].iloc[-1]) * 100
                print(f"\nCurrent Profit Margin: {current_margin:.1f}%")
                
                if current_margin < 10:
                    print("⚠️ Recommendation: Focus on cost optimization and pricing strategies")
                elif current_margin > 20:
                    print("✅ Strong profit margins, consider expansion opportunities")
        
        # Business Health Score
        print("\n3. BUSINESS HEALTH SCORE:")
        print("-" * 75)
        health_score = self.calculate_business_health_score()
        print(f"Overall Business Health Score: {health_score}/100")
        
        # Recommendations
        print("\n4. KEY RECOMMENDATIONS:")
        print("-" * 75)
        self.generate_recommendations(forecast_data)

    def calculate_business_health_score(self):
        """Calculate overall business health score"""
        score = 0
        metrics = 0
        
        if 'current_ratio' in self.ratios:
            cr = self.ratios['current_ratio'].iloc[-1]
            score += min(100, cr * 25) if cr <= 4 else 75
            metrics += 1
        
        if 'debt_to_equity' in self.ratios:
            dte = self.ratios['debt_to_equity'].iloc[-1]
            score += max(0, 100 - (dte * 25)) if dte <= 4 else 0
            metrics += 1
        
        if 'net_profit_margin' in self.ratios:
            npm = self.ratios['net_profit_margin'].iloc[-1]
            score += min(100, npm * 5)
            metrics += 1
        
        if 'roa' in self.ratios:
            roa = self.ratios['roa'].iloc[-1]
            score += min(100, roa * 10)
            metrics += 1
        
        return round(score / metrics if metrics > 0 else 0)

    def generate_recommendations(self, forecast_data=None):
        """Generate business recommendations based on ratios and forecasts"""
        if 'current_ratio' in self.ratios:
            cr = self.ratios['current_ratio'].iloc[-1]
            if cr < 1.0:
                print("- Improve working capital management")
                print("- Consider reducing short-term liabilities")
            elif cr > 3.0:
                print("- Consider investing excess current assets")
        
        if 'debt_to_equity' in self.ratios:
            dte = self.ratios['debt_to_equity'].iloc[-1]
            if dte > 2.0:
                print("- Review debt structure and consider debt consolidation")
                print("- Focus on debt reduction strategies")
        
        if forecast_data and 'revenue' in forecast_data:
            growth_rate = ((forecast_data['revenue']['SARIMA'][11] / 
                          self.df['revenue'].iloc[-1]) - 1) * 100
            
            if growth_rate > 20:
                print("- Prepare for high growth: ensure infrastructure and resources")
            elif growth_rate < 0:
                print("- Develop strategies to reverse negative growth trend")
                print("- Review pricing and marketing strategies")

def main():
    parser = argparse.ArgumentParser(description='Financial Ratio Analysis Tool')
    parser.add_argument('--dataset', type=str, help='Path to the financial metrics CSV file')
    parser.add_argument('--with-forecasts', action='store_true', 
                      help='Include forecasting analysis')
    parser.add_argument('--auto-update', action='store_true',
                      help='Enable automatic updates when data changes')
    parser.add_argument('--update-interval', type=int, default=300,
                      help='Update check interval in seconds (default: 300)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FinancialRatioAnalyzer(args.dataset, auto_update=args.auto_update)
    if args.update_interval:
        analyzer.update_interval = args.update_interval
    
    if analyzer.load_data():
        if args.with_forecasts:
            # Import forecasting module
            import business_forecasting as bf
            
            # Get forecasts
            forecast_results = bf.forecast_financials(
                dataset_path=args.dataset,
                forecast_periods=12,
                output_type='month'
            )
            
            # Analyze with forecasts
            analyzer.analyze_with_forecasts(forecast_results['forecasts'])
        else:
            # Regular ratio analysis
            analyzer.calculate_and_display_ratios()
        
        print("\nGenerating trend visualizations...")
        analyzer.plot_ratio_trends()
        print("Ratio trend plots have been saved to the 'plots' directory.")
        
        try:
            # Keep the program running if auto-update is enabled
            if args.auto_update:
                print("\nPress Ctrl+C to stop monitoring...")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            if analyzer.auto_update:
                analyzer.stop_auto_update()
            print("\nMonitoring stopped.")
    else:
        print("\nError: Could not load or process the dataset.")

if __name__ == "__main__":
    main()
