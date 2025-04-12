import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class FinancialAnalyzer:
    def __init__(self):
        # Load and preprocess data
        self.financial_metrics = pd.read_csv('financial_metrics.csv')
        self.transactions = pd.read_csv('transactions.csv')
        self.users = pd.read_csv('users.csv')
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the datasets"""
        # Convert date columns
        self.financial_metrics['date'] = pd.to_datetime(self.financial_metrics['date'])
        self.transactions['date'] = pd.to_datetime(self.transactions['date'])
        
        # Calculate derived metrics
        self.financial_metrics['profit'] = self.financial_metrics['revenue'] - self.financial_metrics['expenses']
        
        # Data validation
        print("\nData Validation Report:")
        print(f"Financial Metrics Records: {len(self.financial_metrics):,}")
        print(f"Transaction Records: {len(self.transactions):,}")
        print(f"User Records: {len(self.users):,}")
        print("\nDate Range:")
        print(f"Financial Metrics: {self.financial_metrics['date'].min()} to {self.financial_metrics['date'].max()}")
        print(f"Transactions: {self.transactions['date'].min()} to {self.transactions['date'].max()}")
        
    def revenue_analysis(self):
        """Analyze revenue trends and patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Revenue by Business Unit
        sns.boxplot(data=self.financial_metrics, x='business_unit', y='revenue', ax=axes[0,0])
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45, ha='right')
        axes[0,0].set_title('Revenue Distribution by Business Unit')
        axes[0,0].set_ylabel('Revenue')
        
        # Revenue Trend
        monthly_data = self.financial_metrics.groupby(
            pd.Grouper(key='date', freq='ME')
        ).agg({
            'revenue': 'mean',
            'expenses': 'mean'
        }).reset_index()
        
        axes[0,1].plot(monthly_data['date'], monthly_data['revenue'], label='Revenue')
        axes[0,1].plot(monthly_data['date'], monthly_data['expenses'], label='Expenses')
        axes[0,1].set_title('Monthly Average Revenue and Expenses Trend')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend()
        
        # Revenue vs Expenses Correlation
        correlation = self.financial_metrics['revenue'].corr(self.financial_metrics['expenses'])
        axes[1,0].scatter(self.financial_metrics['revenue'], self.financial_metrics['expenses'])
        axes[1,0].set_xlabel('Revenue')
        axes[1,0].set_ylabel('Expenses')
        axes[1,0].set_title(f'Revenue vs Expenses\nCorrelation: {correlation:.2f}')
        
        # Profit Distribution
        sns.histplot(data=self.financial_metrics, x='profit', bins=50, ax=axes[1,1])
        axes[1,1].set_title('Profit Distribution')
        
        plt.tight_layout()
        plt.savefig('revenue_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def transaction_analysis(self):
        """Analyze transaction patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Transaction Types
        tx_types = self.transactions['transaction_type'].value_counts()
        axes[0,0].pie(tx_types.values, labels=tx_types.index, autopct='%1.1f%%')
        axes[0,0].set_title('Transaction Types Distribution')
        
        # Payment Methods
        payment_methods = self.transactions['payment_method'].value_counts()
        sns.barplot(x=payment_methods.index, y=payment_methods.values, ax=axes[0,1])
        axes[0,1].set_title('Payment Methods Distribution')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Transaction Amount Distribution
        sns.histplot(data=self.transactions, x='amount', bins=50, ax=axes[1,0])
        axes[1,0].set_title('Transaction Amount Distribution')
        
        # Top Categories
        top_categories = self.transactions['category'].value_counts().head(10)
        sns.barplot(x=top_categories.values, y=top_categories.index, ax=axes[1,1])
        axes[1,1].set_title('Top 10 Transaction Categories')
        
        plt.tight_layout()
        plt.savefig('transaction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def business_performance(self):
        """Analyze business unit performance"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # ROI by Business Unit
        sns.boxplot(data=self.financial_metrics, x='business_unit', y='roi', ax=axes[0,0])
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45, ha='right')
        axes[0,0].set_title('ROI Distribution by Business Unit')
        
        # Profit Margin by Business Unit
        sns.boxplot(data=self.financial_metrics, x='business_unit', y='profit_margin', ax=axes[0,1])
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45, ha='right')
        axes[0,1].set_title('Profit Margin Distribution by Business Unit')
        
        # Total Revenue and Expenses by Business Unit
        bu_metrics = self.financial_metrics.groupby('business_unit').agg({
            'revenue': 'sum',
            'expenses': 'sum',
            'profit': 'sum'
        }).round(2)
        
        x = np.arange(len(bu_metrics.index))
        width = 0.35
        
        axes[1,0].bar(x - width/2, bu_metrics['revenue'], width, label='Revenue')
        axes[1,0].bar(x + width/2, bu_metrics['expenses'], width, label='Expenses')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(bu_metrics.index, rotation=45, ha='right')
        axes[1,0].set_title('Total Revenue vs Expenses by Business Unit')
        axes[1,0].legend()
        
        # Profit by Business Unit
        sns.barplot(data=self.financial_metrics, x='business_unit', y='profit', ax=axes[1,1])
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45, ha='right')
        axes[1,1].set_title('Total Profit by Business Unit')
        
        plt.tight_layout()
        plt.savefig('business_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_report(self):
        """Generate detailed summary statistics"""
        summary = {
            'Financial Overview': {
                'Total Revenue': f"${self.financial_metrics['revenue'].sum():,.2f}",
                'Total Expenses': f"${self.financial_metrics['expenses'].sum():,.2f}",
                'Total Profit': f"${self.financial_metrics['profit'].sum():,.2f}",
                'Average Profit Margin': f"{self.financial_metrics['profit_margin'].mean():.2f}%",
                'Average ROI': f"{self.financial_metrics['roi'].mean():.2f}%"
            },
            'Transaction Metrics': {
                'Total Transactions': f"{len(self.transactions):,}",
                'Average Transaction Amount': f"${self.transactions['amount'].mean():,.2f}",
                'Most Common Payment Method': self.transactions['payment_method'].mode()[0],
                'Most Common Transaction Type': self.transactions['transaction_type'].mode()[0]
            },
            'Business Unit Performance': {
                'Best Performing (ROI)': self.financial_metrics.groupby('business_unit')['roi'].mean().idxmax(),
                'Best Performing (Profit)': self.financial_metrics.groupby('business_unit')['profit'].sum().idxmax()
            }
        }
        return pd.DataFrame(summary)
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("=== Financial Analysis Report ===\n")
        
        print("Summary Statistics:")
        print(self.generate_summary_report())
        
        # Add temporal analysis
        self.temporal_analysis()
        
        print("\nAnalysis Visualizations have been saved as:")
        print("1. revenue_analysis.png - Revenue trends and distributions")
        print("2. transaction_analysis.png - Transaction patterns and categories")
        print("3. business_performance.png - Business unit performance metrics")
        print("4. temporal_analysis.png - Monthly and yearly trends")
        
        # Generate all visualizations
        self.revenue_analysis()
        self.transaction_analysis()
        self.business_performance()

    def temporal_analysis(self):
        """Analyze monthly and yearly trends"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Monthly Analysis
        monthly_metrics = self.financial_metrics.groupby(
            self.financial_metrics['date'].dt.to_period('M')
        ).agg({
            'revenue': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        # Convert period to datetime for plotting
        monthly_metrics['date'] = monthly_metrics['date'].dt.to_timestamp()
        
        # Monthly Revenue Trend
        axes[0,0].plot(monthly_metrics['date'], monthly_metrics['revenue'], 
                       marker='o', linewidth=2, label='Revenue')
        axes[0,0].set_title('Monthly Revenue Trend')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Revenue ($)')
        plt.setp(axes[0,0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[0,0].grid(True)
        
        # Monthly Profit Trend
        axes[0,1].plot(monthly_metrics['date'], monthly_metrics['profit'], 
                       marker='o', linewidth=2, color='green', label='Profit')
        axes[0,1].set_title('Monthly Profit Trend')
        axes[0,1].set_xlabel('Month')
        axes[0,1].set_ylabel('Profit ($)')
        plt.setp(axes[0,1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[0,1].grid(True)
        
        # Yearly Analysis
        yearly_metrics = self.financial_metrics.groupby(
            self.financial_metrics['date'].dt.year
        ).agg({
            'revenue': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        # Yearly Revenue
        axes[1,0].bar(yearly_metrics['date'], yearly_metrics['revenue'], 
                      color='blue', alpha=0.7)
        axes[1,0].set_title('Yearly Revenue')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Revenue ($)')
        # Add value labels on bars
        for i, v in enumerate(yearly_metrics['revenue']):
            axes[1,0].text(yearly_metrics['date'][i], v, f'${v:,.0f}', 
                          ha='center', va='bottom')
        
        # Yearly Profit
        axes[1,1].bar(yearly_metrics['date'], yearly_metrics['profit'], 
                      color='green', alpha=0.7)
        axes[1,1].set_title('Yearly Profit')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Profit ($)')
        # Add value labels on bars
        for i, v in enumerate(yearly_metrics['profit']):
            axes[1,1].text(yearly_metrics['date'][i], v, f'${v:,.0f}', 
                          ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\nYearly Summary:")
        print(yearly_metrics.set_index('date').round(2))
        
        # Calculate year-over-year growth
        yearly_metrics['revenue_growth'] = yearly_metrics['revenue'].pct_change() * 100
        yearly_metrics['profit_growth'] = yearly_metrics['profit'].pct_change() * 100
        
        print("\nYear-over-Year Growth (%):")
        print(yearly_metrics[['date', 'revenue_growth', 'profit_growth']].set_index('date').round(2))

def main():
    analyzer = FinancialAnalyzer()
    analyzer.generate_report()

if __name__ == "__main__":
    main() 