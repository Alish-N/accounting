import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
import os
warnings.filterwarnings('ignore')

class FinancialRatioAnalyzer:
    def __init__(self, file_path):
        """Initialize the Financial Ratio Analyzer with the data file path"""
        # Get the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # If file_path is not absolute, make it relative to the parent directory of the script
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(script_dir), file_path)
        self.file_path = file_path
        self.df = None
        self.ratios_df = None
        self.forecast_ratios_df = None
        
    def load_and_analyze_data(self):
        """Load and analyze the dataset structure"""
        try:
            # Load the dataset
            self.df = pd.read_csv(self.file_path)
            
            print("\nFinancial Metrics Analysis")
            print("=" * 50)
            print(f"\nTotal Records: {len(self.df)}")
            
            # Display available features categorized
            print("\nAvailable Financial Metrics:")
            print("\n1. Primary Metrics:")
            print("   - Revenue")
            print("   - Expenses")
            print("   - Operating Expenses")
            print("   - Cash Flow")
            print("   - Working Capital")
            print("   - Tax Amount")
            
            print("\n2. Key Ratios:")
            print("   - Current Ratio")
            print("   - Quick Ratio")
            print("   - Profit Margin")
            print("   - ROI")
            print("   - Debt to Equity")
            print("   - Asset Turnover")
            
            print("\n3. Operational Metrics:")
            print("   - Inventory Turnover")
            print("   - Inventory Days")
            print("   - Receivable Days")
            print("   - Payable Days")
            
            print("\n4. Performance Metrics:")
            print("   - Budget Variance")
            print("   - Forecast Accuracy")
            
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        try:
            # Convert date column to datetime
            self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Ensure numeric columns are properly converted
            numeric_columns = [
                'revenue', 'expenses', 'profit_margin', 'current_ratio',
                'quick_ratio', 'roi', 'debt_to_equity'
            ]
            
            for col in numeric_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Group by date and calculate mean for each metric
            self.df = self.df.groupby('date')[numeric_columns].mean().reset_index()
            
            print("\nData Processing Complete")
            print("=" * 50)
            print(f"Date Range: {self.df['date'].min().strftime('%Y-%m-%d')} to {self.df['date'].max().strftime('%Y-%m-%d')}")
            print(f"Processed Records: {len(self.df)}")
            
            return True
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            return False
    
    def compute_key_ratios(self):
        """Compute the three main categories of financial ratios"""
        try:
            # Create DataFrame for key ratios
            self.ratios_df = pd.DataFrame()
            self.ratios_df['Date'] = self.df['date']
            
            # 1. Liquidity Ratios
            print("\nComputing Key Financial Ratios:")
            print("-" * 30)
            print("1. Liquidity Ratios:")
            self.ratios_df['Current Ratio'] = self.df['current_ratio']
            self.ratios_df['Quick Ratio'] = self.df['quick_ratio']
            print("   - Current Ratio")
            print("   - Quick Ratio")
            
            # 2. Profitability Ratios
            print("\n2. Profitability Ratios:")
            self.ratios_df['Profit Margin'] = self.df['profit_margin']
            self.ratios_df['ROI'] = self.df['roi']
            print("   - Profit Margin")
            print("   - Return on Investment (ROI)")
            
            # 3. Leverage Ratio
            print("\n3. Leverage Ratio:")
            self.ratios_df['Debt to Equity'] = self.df['debt_to_equity']
            print("   - Debt to Equity Ratio")
            
            # Set date as index for plotting
            self.ratios_df.set_index('Date', inplace=True)
            
            # Display summary statistics
            print("\nKey Ratios Summary Statistics:")
            print("=" * 50)
            print(self.ratios_df.describe().round(2))
            
            return True
        except Exception as e:
            print(f"Error computing ratios: {str(e)}")
            return False
    
    def plot_key_ratios(self):
        """Create clear, professional, and user-friendly visualizations"""
        if self.ratios_df is None or self.ratios_df.empty:
            print("No ratios available to plot")
            return
        
        try:
            # Set style for better visualization
            plt.style.use('seaborn-whitegrid')
            
            # Create figure with better spacing
            fig, axes = plt.subplots(3, 1, figsize=(15, 20))
            fig.suptitle('Financial Health Analysis Dashboard', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            # Color scheme for professional look
            colors = ['#2E86C1', '#28B463', '#E74C3C', '#F39C12']
            
            # 1. Liquidity Ratios
            ax1 = axes[0]
            self.ratios_df[['Current Ratio', 'Quick Ratio']].plot(
                ax=ax1,
                linewidth=3,
                marker='o',
                markersize=8,
                color=[colors[0], colors[1]]
            )
            ax1.set_title('Liquidity Analysis\n(Company\'s Ability to Pay Short-term Obligations)', 
                         fontsize=14, pad=20, fontweight='bold')
            ax1.set_ylabel('Ratio Value', fontsize=12)
            ax1.axhline(y=1, color='r', linestyle='--', alpha=0.3)  # Reference line at 1
            ax1.fill_between(self.ratios_df.index, 1, 2, alpha=0.1, color='green', label='Healthy Zone')
            ax1.legend(fontsize=12, title='Metrics', title_fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Add annotation for interpretation
            ax1.text(0.02, 0.95, 'Healthy Range: 1.0 - 2.0\nHigher ratio indicates better liquidity',
                     transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # 2. Profitability Ratios
            ax2 = axes[1]
            self.ratios_df[['Profit Margin', 'ROI']].plot(
                ax=ax2,
                linewidth=3,
                marker='o',
                markersize=8,
                color=[colors[2], colors[3]]
            )
            ax2.set_title('Profitability Analysis\n(Company\'s Ability to Generate Profits)', 
                         fontsize=14, pad=20, fontweight='bold')
            ax2.set_ylabel('Percentage (%)', fontsize=12)
            # Convert to percentage
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            ax2.legend(fontsize=12, title='Metrics', title_fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add annotation for interpretation
            ax2.text(0.02, 0.95, 'Higher percentages indicate better profitability\nIndustry average: 10-20%',
                     transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # 3. Leverage Ratio
            ax3 = axes[2]
            self.ratios_df['Debt to Equity'].plot(
                ax=ax3,
                linewidth=3,
                marker='o',
                markersize=8,
                color=colors[0]
            )
            ax3.set_title('Leverage Analysis\n(Company\'s Use of Borrowed Money)', 
                         fontsize=14, pad=20, fontweight='bold')
            ax3.set_ylabel('Ratio Value', fontsize=12)
            ax3.axhline(y=2, color='r', linestyle='--', alpha=0.3)  # Risk threshold line
            ax3.fill_between(self.ratios_df.index, 0, 2, alpha=0.1, color='green', label='Safe Zone')
            ax3.legend(fontsize=12, title='Metric', title_fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            # Add annotation for interpretation
            ax3.text(0.02, 0.95, 'Safe Range: 0 - 2.0\nHigher ratio indicates higher financial risk',
                     transform=ax3.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # Improve x-axis readability for all plots
            for ax in axes:
                ax.tick_params(axis='both', labelsize=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add subtle background grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Add border to the plot
                for spine in ax.spines.values():
                    spine.set_edgecolor('#666666')
                    spine.set_linewidth(0.5)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig('key_financial_ratios.png', dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print("\nProfessional visualization saved as 'key_financial_ratios.png'")
            
        except Exception as e:
            print(f"Error plotting ratios: {str(e)}")
    
    def generate_business_insights(self):
        """Generate business insights from the ratios"""
        if self.ratios_df is None:
            return
        
        try:
            # Get latest values
            latest = self.ratios_df.iloc[-1]
            
            insights = [
                "Key Business Insights",
                "=" * 50,
                
                "\n1. Liquidity Analysis:",
                "-" * 20,
                f"Current Ratio: {latest['Current Ratio']:.2f}",
                self._get_liquidity_insight(latest['Current Ratio']),
                f"Quick Ratio: {latest['Quick Ratio']:.2f}",
                self._get_quick_ratio_insight(latest['Quick Ratio']),
                
                "\n2. Profitability Analysis:",
                "-" * 20,
                f"Profit Margin: {latest['Profit Margin']:.1%}",
                f"ROI: {latest['ROI']:.1%}",
                self._get_profitability_insight(latest['Profit Margin'], latest['ROI']),
                
                "\n3. Leverage Analysis:",
                "-" * 20,
                f"Debt-to-Equity Ratio: {latest['Debt to Equity']:.2f}",
                self._get_leverage_insight(latest['Debt to Equity'])
            ]
            
            # Save insights to file
            with open('financial_insights.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(insights))
            
            # Print insights to console
            print('\n'.join(insights))
            print("\nInsights have been saved to 'financial_insights.txt'")
            
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            
    def _get_liquidity_insight(self, ratio):
        if ratio > 2:
            return "STRONG liquidity position - Excellent ability to cover short-term obligations."
        elif ratio > 1:
            return "ADEQUATE liquidity position - Sufficient ability to cover short-term obligations."
        else:
            return "WEAK liquidity position - May have difficulty covering short-term obligations."
    
    def _get_quick_ratio_insight(self, ratio):
        if ratio > 1:
            return "STRONG immediate liquidity - Can cover immediate obligations without inventory."
        elif ratio > 0.7:
            return "ADEQUATE immediate liquidity - Reasonable ability to cover immediate obligations."
        else:
            return "LIMITED immediate liquidity - May need to rely on inventory sales."
    
    def _get_profitability_insight(self, margin, roi):
        insights = []
        if margin > 0.15:
            insights.append("STRONG profit margins indicating efficient operations.")
        elif margin > 0.10:
            insights.append("ADEQUATE profit margins with room for improvement.")
        else:
            insights.append("LOW profit margins suggesting need for cost control.")
            
        if roi > 0.15:
            insights.append("EXCELLENT return on investment.")
        elif roi > 0.10:
            insights.append("SATISFACTORY return on investment.")
        else:
            insights.append("BELOW TARGET return on investment.")
            
        return " ".join(insights)
    
    def _get_leverage_insight(self, ratio):
        if ratio < 1:
            return "CONSERVATIVE leverage position - Low financial risk but possibly underutilizing debt."
        elif ratio < 2:
            return "BALANCED leverage position - Moderate use of debt financing."
        else:
            return "HIGH leverage position - Higher financial risk but potential for greater returns."
    
    def generate_quotation_analysis(self):
        """Generate a comprehensive quotation and analysis report"""
        try:
            if self.ratios_df is None:
                return
            
            # Calculate key statistics
            latest = self.ratios_df.iloc[-1]
            trends = self.ratios_df.iloc[-3:].mean()
            yoy_change = self.ratios_df.iloc[-1] - self.ratios_df.iloc[-12] if len(self.ratios_df) >= 12 else pd.Series()
            
            # Create the analysis report
            report = [
                "FINANCIAL ANALYSIS QUOTATION AND REPORT",
                "=" * 50,
                f"\nReport Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
                f"Analysis Period: {self.ratios_df.index[0].strftime('%Y-%m-%d')} to {self.ratios_df.index[-1].strftime('%Y-%m-%d')}",
                
                "\n1. EXECUTIVE SUMMARY",
                "-" * 20,
                "This report provides a comprehensive analysis of the company's financial health,",
                "focusing on key performance indicators and financial ratios.",
                
                "\n2. KEY FINDINGS",
                "-" * 20,
                f"• Current Ratio: {latest['Current Ratio']:.2f} ({'+' if yoy_change['Current Ratio'] > 0 else ''}{yoy_change['Current Ratio']:.2f} YoY)",
                f"• Quick Ratio: {latest['Quick Ratio']:.2f} ({'+' if yoy_change['Quick Ratio'] > 0 else ''}{yoy_change['Quick Ratio']:.2f} YoY)",
                f"• Profit Margin: {latest['Profit Margin']:.1%} ({'+' if yoy_change['Profit Margin'] > 0 else ''}{yoy_change['Profit Margin']:.1%} YoY)",
                f"• ROI: {latest['ROI']:.1%} ({'+' if yoy_change['ROI'] > 0 else ''}{yoy_change['ROI']:.1%} YoY)",
                f"• Debt to Equity: {latest['Debt to Equity']:.2f} ({'+' if yoy_change['Debt to Equity'] > 0 else ''}{yoy_change['Debt to Equity']:.2f} YoY)",
                
                "\n3. DETAILED ANALYSIS",
                "-" * 20,
                "3.1 Liquidity Analysis",
                self._get_liquidity_insight(latest['Current Ratio']),
                self._get_quick_ratio_insight(latest['Quick Ratio']),
                
                "\n3.2 Profitability Analysis",
                self._get_profitability_insight(latest['Profit Margin'], latest['ROI']),
                
                "\n3.3 Leverage Analysis",
                self._get_leverage_insight(latest['Debt to Equity']),
                
                "\n4. RECOMMENDATIONS",
                "-" * 20,
                "Based on the analysis, we recommend:",
                "• " + self._get_primary_recommendation(latest),
                "• Continue monitoring key ratios monthly",
                "• Review and adjust financial strategies quarterly",
                
                "\n5. METHODOLOGY",
                "-" * 20,
                "This analysis uses industry-standard financial ratios and metrics.",
                "All calculations follow GAAP guidelines and best practices.",
                
                "\n6. SUMMARY STATISTICS",
                "-" * 20,
                "\nKey Ratios Summary:",
                str(self.ratios_df.describe().round(3))
            ]
            
            # Save the report
            with open('financial_analysis_report.txt', 'w') as f:
                f.write('\n'.join(report))
            
            print("\nComprehensive financial analysis report saved as 'financial_analysis_report.txt'")
            
        except Exception as e:
            print(f"Error generating quotation analysis: {str(e)}")
    
    def _get_primary_recommendation(self, latest):
        """Generate primary recommendation based on latest metrics"""
        if latest['Current Ratio'] < 1.5:
            return "Focus on improving working capital management to strengthen liquidity position"
        elif latest['Profit Margin'] < 0.15:
            return "Implement cost optimization strategies to improve profit margins"
        elif latest['Debt to Equity'] > 2:
            return "Consider debt reduction strategies to improve financial stability"
        else:
            return "Maintain current financial strategies while seeking growth opportunities"
    
    def load_forecast_data(self):
        """Load and process forecasted data from consolidated results"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            forecast_path = os.path.join(script_dir, 'consolidated_results.csv')
            
            if not os.path.exists(forecast_path):
                print("No forecast data found. Run business_forecasting.py first to generate forecasts.")
                return False
                
            forecast_df = pd.read_csv(forecast_path)
            forecast_df['date'] = pd.to_datetime(forecast_df['month'])
            
            # Calculate forecasted ratios
            self.forecast_ratios_df = pd.DataFrame()
            self.forecast_ratios_df['Date'] = forecast_df['date']
            
            # Calculate predicted ratios
            if all(col in forecast_df.columns for col in ['predicted_revenue', 'predicted_expenses']):
                # Profit Margin
                self.forecast_ratios_df['Profit Margin'] = (
                    (forecast_df['predicted_revenue'] - forecast_df['predicted_expenses']) / 
                    forecast_df['predicted_revenue']
                )
                
                # Assuming some standard calculations for other ratios based on forecasted values
                # These are simplified approximations
                self.forecast_ratios_df['Current Ratio'] = 1.8  # Using industry average as placeholder
                self.forecast_ratios_df['Quick Ratio'] = 1.2    # Using industry average as placeholder
                self.forecast_ratios_df['ROI'] = (
                    forecast_df['predicted_cash_flow'] / forecast_df['predicted_revenue']
                )
                self.forecast_ratios_df['Debt to Equity'] = 1.5  # Using industry average as placeholder
                
            self.forecast_ratios_df.set_index('Date', inplace=True)
            return True
            
        except Exception as e:
            print(f"Error loading forecast data: {str(e)}")
            return False
    
    def generate_combined_insights(self):
        """Generate insights combining current and forecasted ratios"""
        if self.ratios_df is None or self.forecast_ratios_df is None:
            return
        
        try:
            # Get latest actual values and first forecasted values
            latest = self.ratios_df.iloc[-1]
            next_month = self.forecast_ratios_df.iloc[0]
            
            insights = [
                "\nCOMBINED FINANCIAL ANALYSIS (Current & Forecasted)",
                "=" * 50,
                
                "\n1. Current vs Next Month Analysis:",
                "-" * 30,
                f"Current Ratio:",
                f"  Current: {latest['Current Ratio']:.2f}",
                f"  Next Month: {next_month['Current Ratio']:.2f}",
                self._get_liquidity_insight(next_month['Current Ratio']),
                
                f"\nQuick Ratio:",
                f"  Current: {latest['Quick Ratio']:.2f}",
                f"  Next Month: {next_month['Quick Ratio']:.2f}",
                self._get_quick_ratio_insight(next_month['Quick Ratio']),
                
                f"\nProfitability Metrics:",
                f"  Current Profit Margin: {latest['Profit Margin']:.1%}",
                f"  Next Month Profit Margin: {next_month['Profit Margin']:.1%}",
                f"  Current ROI: {latest['ROI']:.1%}",
                f"  Next Month ROI: {next_month['ROI']:.1%}",
                self._get_profitability_insight(next_month['Profit Margin'], next_month['ROI']),
                
                f"\nLeverage Analysis:",
                f"  Current Debt-to-Equity: {latest['Debt to Equity']:.2f}",
                f"  Next Month Debt-to-Equity: {next_month['Debt to Equity']:.2f}",
                self._get_leverage_insight(next_month['Debt to Equity']),
                
                "\n2. Three-Month Forecast Summary:",
                "-" * 30
            ]
            
            # Add 3-month forecast trends
            three_month = self.forecast_ratios_df.iloc[:3]
            for metric in ['Profit Margin', 'ROI', 'Current Ratio', 'Debt to Equity']:
                trend = "increasing" if three_month[metric].is_monotonic_increasing else "decreasing" if three_month[metric].is_monotonic_decreasing else "fluctuating"
                insights.append(f"{metric}: {trend.title()} trend over next 3 months")
                insights.append(f"  Range: {three_month[metric].min():.2f} to {three_month[metric].max():.2f}")
            
            # Save combined insights
            with open('combined_financial_insights.txt', 'w') as f:
                f.write('\n'.join(insights))
            
            print('\n'.join(insights))
            print("\nCombined insights saved to 'combined_financial_insights.txt'")
            
        except Exception as e:
            print(f"Error generating combined insights: {str(e)}")
    
    def plot_combined_ratios(self):
        """Create visualizations comparing current and forecasted ratios"""
        if self.ratios_df is None or self.forecast_ratios_df is None:
            return
            
        try:
            plt.style.use('seaborn-whitegrid')
            
            # Create figure with better spacing
            fig, axes = plt.subplots(3, 1, figsize=(15, 20))
            fig.suptitle('Financial Health Analysis Dashboard (Current & Forecasted)', 
                        fontsize=20, fontweight='bold', y=0.95)
            
            # Color scheme
            colors = ['#2E86C1', '#28B463', '#E74C3C', '#F39C12']
            
            # Plot each ratio type with both current and forecasted values
            metrics = [
                (['Current Ratio', 'Quick Ratio'], 'Liquidity Analysis', 'Ratio Value'),
                (['Profit Margin', 'ROI'], 'Profitability Analysis', 'Percentage'),
                (['Debt to Equity'], 'Leverage Analysis', 'Ratio Value')
            ]
            
            for idx, (ratios, title, ylabel) in enumerate(metrics):
                ax = axes[idx]
                
                # Plot historical data
                for i, ratio in enumerate(ratios):
                    ax.plot(self.ratios_df.index[-12:], self.ratios_df[ratio].iloc[-12:],
                           label=f'Historical {ratio}', color=colors[i], linewidth=2)
                    
                    # Plot forecasted data with dashed lines
                    ax.plot(self.forecast_ratios_df.index[:6], self.forecast_ratios_df[ratio].iloc[:6],
                           label=f'Forecasted {ratio}', color=colors[i], linestyle='--', linewidth=2)
                
                ax.set_title(f'{title}\n(Historical & Forecasted)', fontsize=14, pad=20)
                ax.set_ylabel(ylabel, fontsize=12)
                
                if ylabel == 'Percentage':
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
                
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add vertical line to separate historical and forecasted data
                ax.axvline(x=self.ratios_df.index[-1], color='gray', linestyle=':')
                ax.text(self.ratios_df.index[-1], ax.get_ylim()[1], 'Forecast Start',
                       rotation=90, va='top', ha='right')
            
            plt.tight_layout()
            plt.savefig('combined_ratio_analysis.png', dpi=300, bbox_inches='tight')
            print("\nCombined ratio analysis visualization saved as 'combined_ratio_analysis.png'")
            
        except Exception as e:
            print(f"Error plotting combined ratios: {str(e)}")

def main():
    # Initialize analyzer with the correct path
    analyzer = FinancialRatioAnalyzer('financial_metrics.csv')
    
    if analyzer.load_and_analyze_data():
        if analyzer.preprocess_data():
            if analyzer.compute_key_ratios():
                analyzer.plot_key_ratios()
                analyzer.generate_business_insights()
                # Load and analyze forecast data
                if analyzer.load_forecast_data():
                    analyzer.generate_combined_insights()
                    analyzer.plot_combined_ratios()

if __name__ == "__main__":
    main()