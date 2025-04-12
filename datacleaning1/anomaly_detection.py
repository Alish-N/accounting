import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from scipy.stats import median_abs_deviation
import os
from datetime import datetime, timedelta
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

class TransactionAnomalyDetector:
    def __init__(self):
        """Initialize the anomaly detector"""
        print("Initializing Anomaly Detection System...")
        self._create_directories()
        self.load_data()
        self._engineer_features()
        self.models = {}
        self.scaler = None
        self.feature_importance = {}

    def _create_directories(self):
        """Create necessary directories"""
        directories = ['anomaly_results', 'models', 'evaluation_results']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def load_data(self):
        """Load cleaned transaction data"""
        try:
            self.transactions = pd.read_csv('cleaned_data/cleaned_transactions.csv')
            self.transactions['date'] = pd.to_datetime(self.transactions['date'])
            print(f"Loaded {len(self.transactions)} transactions\n")
            
            # Try to load user data for behavioral analysis
            try:
                self.users = pd.read_csv('cleaned_data/cleaned_users.csv')
                print(f"Loaded {len(self.users)} users for behavioral analysis")
                self.has_user_data = True
            except:
                print("User data not available. Behavioral analysis will be limited.")
                self.has_user_data = False
                
            self._analyze_data_distribution()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _analyze_data_distribution(self):
        """Analyze data distribution using robust statistics"""
        numeric_cols = [
            'amount', 'total_tax', 'total_tax_calculated', 
            'tax_discrepancy'
        ]
        
        print("Data Distribution Summary:")
        self.distribution_stats = {}
        
        for col in numeric_cols:
            if col not in self.transactions.columns:
                print(f"Column {col} not found in data. Skipping analysis.")
                continue
                
            data = self.transactions[col].fillna(self.transactions[col].median())
            
            # Calculate robust statistics
            median = np.median(data)
            mad = median_abs_deviation(data, scale='normal')
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            
            # Calculate outliers using IQR method
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = np.sum((data < lower_bound) | (data > upper_bound))
            
            # Store statistics for later use
            self.distribution_stats[col] = {
                'median': median,
                'mad': mad,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_percentage': outliers/len(data)
            }
            
            print(f"\n{col.upper()} ANALYSIS:")
            print(f"Median: ${median:,.2f}")
            print(f"MAD: ${mad:,.2f}")
            print(f"IQR: ${iqr:,.2f}")
            print(f"Outliers: {outliers} ({(outliers/len(data)*100):.2f}%)")

    def _engineer_features(self):
        """Engineer features for anomaly detection"""
        print("\nEngineering features for anomaly detection...")
        
        # 1. Basic time-based features
        self.transactions['hour'] = self.transactions['date'].dt.hour
        self.transactions['day_of_week'] = self.transactions['date'].dt.dayofweek
        self.transactions['month'] = self.transactions['date'].dt.month
        self.transactions['is_weekend'] = self.transactions['day_of_week'].isin([5,6]).astype(int)
        self.transactions['is_business_hours'] = ((self.transactions['hour'] >= 9) & 
                                                 (self.transactions['hour'] <= 17)).astype(int)
        
        # 2. Advanced temporal features
        if 'created_by' in self.transactions.columns:
            # Sort by user and date
            self.transactions = self.transactions.sort_values(['created_by', 'date'])
            
            # Calculate time difference between consecutive transactions
            self.transactions['prev_tx_date'] = self.transactions.groupby('created_by')['date'].shift(1)
            self.transactions['time_since_last_tx'] = (self.transactions['date'] - 
                                                      self.transactions['prev_tx_date']).dt.total_seconds() / 3600
            
            # Fill NaN values
            self.transactions['time_since_last_tx'] = self.transactions['time_since_last_tx'].fillna(0)
            
            # Calculate transaction velocity and acceleration
            self.transactions['tx_date'] = self.transactions['date'].dt.date
            
            def calculate_velocity_and_acceleration(user_df):
                velocities = []
                accelerations = []
                for i, row in user_df.iterrows():
                    current_date = row['tx_date']
                    week_before = current_date - timedelta(days=7)
                    two_weeks_before = current_date - timedelta(days=14)
                    
                    # Calculate velocity (transactions per day)
                    past_week_count = len(user_df[(user_df['tx_date'] >= week_before) & 
                                                 (user_df['tx_date'] < current_date)])
                    velocity = past_week_count / 7
                    
                    # Calculate acceleration (change in velocity)
                    past_two_weeks_count = len(user_df[(user_df['tx_date'] >= two_weeks_before) & 
                                                      (user_df['tx_date'] < week_before)])
                    previous_velocity = past_two_weeks_count / 7
                    acceleration = velocity - previous_velocity
                    
                    velocities.append(velocity)
                    accelerations.append(acceleration)
                return velocities, accelerations
            
            # Apply calculations by user
            velocity_list = []
            acceleration_list = []
            for user, group in self.transactions.groupby('created_by'):
                user_velocities, user_accelerations = calculate_velocity_and_acceleration(group)
                velocity_list.extend(user_velocities)
                acceleration_list.extend(user_accelerations)
            
            self.transactions['tx_velocity'] = velocity_list
            self.transactions['tx_acceleration'] = acceleration_list
        
        # 3. Amount-based features
        if 'amount' in self.transactions.columns:
            # Calculate Z-score for amount using robust statistics
            median = self.distribution_stats.get('amount', {}).get('median', 
                                                                 self.transactions['amount'].median())
            mad = self.distribution_stats.get('amount', {}).get('mad', 
                                                              median_abs_deviation(self.transactions['amount']))
            
            self.transactions['amount_zscore'] = np.abs(self.transactions['amount'] - median) / mad
            
            # Calculate amount deviation from user average
            if 'created_by' in self.transactions.columns:
                user_stats = self.transactions.groupby('created_by')['amount'].agg(['mean', 'std']).reset_index()
                user_stats.columns = ['created_by', 'user_avg_amount', 'user_std_amount']
                
                median_std = user_stats['user_std_amount'].median()
                user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(median_std)
                
                self.transactions = pd.merge(self.transactions, user_stats, on='created_by', how='left')
                
                self.transactions['amount_user_deviation'] = np.abs(
                    (self.transactions['amount'] - self.transactions['user_avg_amount']) / 
                    self.transactions['user_std_amount']
                )
                
                self.transactions['amount_user_deviation'] = self.transactions['amount_user_deviation'].fillna(0)
        
        # 4. Behavioral patterns
        if self.has_user_data and 'created_by' in self.transactions.columns:
            self.transactions = pd.merge(self.transactions, 
                                        self.users[['user_id', 'joining_date', 'department']], 
                                        left_on='created_by', 
                                        right_on='user_id', 
                                        how='left')
            
            if 'joining_date' in self.transactions.columns:
                self.transactions['joining_date'] = pd.to_datetime(self.transactions['joining_date'])
                self.transactions['user_experience_days'] = (
                    self.transactions['date'] - self.transactions['joining_date']
                ).dt.days
                
                self.transactions['user_experience_days'] = self.transactions['user_experience_days'].fillna(0)
        
        # 5. Tax-related features
        if all(col in self.transactions.columns for col in ['total_tax', 'amount']):
            self.transactions['tax_ratio'] = self.transactions['total_tax'] / self.transactions['amount']
            self.transactions['tax_ratio'] = self.transactions['tax_ratio'].fillna(0)
            
            # Calculate tax discrepancy features
            if 'tax_discrepancy' in self.transactions.columns:
                self.transactions['tax_discrepancy_ratio'] = self.transactions['tax_discrepancy'] / self.transactions['amount']
                self.transactions['tax_discrepancy_ratio'] = self.transactions['tax_discrepancy_ratio'].fillna(0)
        
        print("Feature engineering complete.")

    def detect_anomalies(self):
        """Detect anomalies using multiple methods"""
        print("\nDetecting anomalies using multiple methods...")
        
        # Define core features without duplicates
        core_features = [
            'amount', 'total_tax', 'tax_discrepancy',
            'hour', 'day_of_week', 'month', 'time_since_last_tx',
            'amount_zscore', 'user_avg_amount', 'user_std_amount',
            'tax_ratio'
        ]
        
        # Create feature matrix
        X = self.transactions[core_features].copy()
        
        # Handle missing values with robust imputation
        for col in X.columns:
            if X[col].dtype.kind in 'fc':
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        
        # Scale features using robust scaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate adaptive contamination with balanced approach (6-8% range)
        contamination_metrics = []
        
        if 'tax_discrepancy' in self.transactions.columns:
            tax_mad_scores = np.abs(self.transactions['tax_discrepancy'] - 
                                  np.median(self.transactions['tax_discrepancy'])) / \
                            median_abs_deviation(self.transactions['tax_discrepancy'], nan_policy='omit')
            contamination_metrics.append(np.percentile(tax_mad_scores, 75) / 100)  # Changed from 85 to 75
        
        if 'amount' in self.transactions.columns:
            amount_mad_scores = np.abs(self.transactions['amount'] - 
                                     np.median(self.transactions['amount'])) / \
                              median_abs_deviation(self.transactions['amount'], nan_policy='omit')
            contamination_metrics.append(np.percentile(amount_mad_scores, 75) / 100)  # Changed from 85 to 75
        
        if contamination_metrics:
            # Set contamination rate between 6-8%
            contamination = float(max(0.06, min(0.08, np.mean(contamination_metrics))))
        else:
            contamination = 0.07  # Default to 7% if no metrics available
        
        print(f"Using balanced contamination rate: {contamination:.4f}")
        
        # Initialize detectors with optimized parameters
        detectors = {
            'IsolationForest': IsolationForest(
                contamination=contamination,
                n_estimators=200,  # Increased from 150
                max_samples='auto',
                random_state=42
            ),
            'EllipticEnvelope': EllipticEnvelope(
                contamination=contamination,
                random_state=42,
                support_fraction=0.7  # Decreased from 0.8 for more sensitivity
            ),
            'LocalOutlierFactor': LocalOutlierFactor(
                n_neighbors=15,  # Decreased from 20 for more sensitivity
                contamination=contamination,
                novelty=False
            ),
            'OneClassSVM': OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma='auto'  # Changed from 'scale' to 'auto'
            )
        }
        
        # Train and evaluate each detector
        results = {}
        for name, detector in detectors.items():
            try:
                print(f"\nTraining {name}...")
                
                if name == 'LocalOutlierFactor':
                    pred = detector.fit_predict(X_scaled)
                else:
                    detector.fit(X_scaled)
                    pred = detector.predict(X_scaled)
                
                # Store model
                self.models[name] = detector
                
                # Calculate anomaly scores
                if hasattr(detector, 'score_samples'):
                    scores = detector.score_samples(X_scaled)
                else:
                    scores = np.zeros(len(X_scaled))
                
                # Store results
                results[name] = {
                    'predictions': pred,
                    'scores': scores,
                    'anomaly_count': sum(pred == -1)
                }
                
                print(f"{name} found {results[name]['anomaly_count']} anomalies")
                
            except Exception as e:
                print(f"Warning: {name} failed with error: {str(e)}")
                continue
        
        # Save models
        for name, model in self.models.items():
            try:
                joblib.dump(model, f'models/{name.lower()}_model.joblib')
            except Exception as e:
                print(f"Warning: Could not save {name} model: {str(e)}")
        
        # Combine predictions using business-rule based weighted voting
        anomaly_scores = np.zeros(len(self.transactions))
        total_weight = 0
        
        # Business rule based weights:
        # - IsolationForest: Higher weight for amount-based anomalies
        # - EllipticEnvelope: Higher weight for tax-related anomalies
        # - LocalOutlierFactor: Higher weight for temporal anomalies
        # - OneClassSVM: Balanced weight for overall anomalies
        detector_weights = {
            'IsolationForest': 1.5,  # Increased weight for amount-based detection
            'EllipticEnvelope': 1.3,  # Increased weight for tax-related detection
            'LocalOutlierFactor': 1.2,  # Increased weight for temporal patterns
            'OneClassSVM': 1.0  # Balanced weight
        }
        
        for name, result in results.items():
            if name in detector_weights:
                weight = detector_weights[name]
                anomaly_scores += weight * (result['predictions'] == -1).astype(float)
                total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            anomaly_scores /= total_weight
        
        # Calculate threshold using adjusted MAD
        score_mad = median_abs_deviation(anomaly_scores, scale='normal', nan_policy='omit')
        score_median = np.median(anomaly_scores)
        
        # Adaptive threshold with more balanced approach
        if 'amount_zscore' in self.transactions.columns:
            avg_amount_zscore = self.transactions['amount_zscore'].mean()
            threshold_factor = min(1.8, max(1.0, 1.3 + (avg_amount_zscore / 20)))  # Adjusted range
        else:
            threshold_factor = 1.3  # Decreased from 1.5
        
        threshold = score_median + (threshold_factor * score_mad)
        threshold = min(0.45, max(0.15, threshold))  # Adjusted bounds
        
        print(f"Using balanced anomaly threshold: {threshold:.4f}")
        
        # Flag anomalies with adjusted threshold
        self.transactions['is_anomaly'] = anomaly_scores > threshold
        self.transactions['anomaly_score'] = anomaly_scores
        
        # Add individual model predictions
        for name in results:
            self.transactions[f'anomaly_{name.lower()}'] = (results[name]['predictions'] == -1).astype(int)
        
        # Save results
        self._save_results()
        
        return self.transactions

    def _save_results(self):
        """Save detection results and evaluation metrics"""
        try:
            # Save results
            results_df = self.transactions.copy()
            
            # Define core columns that should always be included
            core_columns = [
                'transaction_id', 'date', 'transaction_type', 'account_type',
                'amount', 'currency', 'payment_method', 'category', 'description',
                'party_name', 'gst_number', 'status', 'created_by', 'approver'
            ]
            
            # Remove duplicate columns
            duplicate_columns = []
            for col in results_df.columns:
                if col.endswith('_x') or col.endswith('_y'):
                    base_col = col[:-2]  # Remove _x or _y suffix
                    if base_col in results_df.columns:
                        duplicate_columns.append(col)
            
            # Drop duplicate columns
            results_df = results_df.drop(columns=duplicate_columns)
            
            # Remove redundant features
            redundant_features = [
                'total_tax_calculated',  # Redundant with total_tax
                'tax_discrepancy_ratio',  # Redundant with tax_discrepancy
                'tx_acceleration',  # Redundant with tx_velocity
                'amount_user_deviation',  # Redundant with user_avg_amount and user_std_amount
                'is_weekend',  # Redundant with day_of_week
                'is_business_hours',  # Redundant with hour
                'anomaly_isolationforest',  # Individual model outputs
                'anomaly_ellipticenvelope',
                'anomaly_localoutlierfactor',
                'anomaly_oneclasssvm'
            ]
            
            # Remove redundant features if they exist
            redundant_features = [f for f in redundant_features if f in results_df.columns]
            results_df = results_df.drop(columns=redundant_features)
            
            # Remove duplicate anomaly_score if it exists
            if 'anomaly_score' in results_df.columns:
                # Keep only the first occurrence
                anomaly_score_cols = [col for col in results_df.columns if col == 'anomaly_score']
                if len(anomaly_score_cols) > 1:
                    results_df = results_df.drop(columns=anomaly_score_cols[1:])
            
            # Organize remaining columns
            key_columns = [col for col in core_columns if col in results_df.columns]
            
            # Get feature columns (excluding core and anomaly columns)
            feature_columns = [col for col in results_df.columns 
                             if col not in key_columns 
                             and not col.startswith('anomaly_') 
                             and col not in ['is_anomaly', 'anomaly_score']
                             and not col.endswith('_x') 
                             and not col.endswith('_y')]
            
            # Get anomaly columns (only keep is_anomaly and anomaly_score)
            anomaly_columns = ['is_anomaly', 'anomaly_score']
            
            # Combine all columns in the desired order
            all_columns = (
                key_columns +
                feature_columns +
                anomaly_columns
            )
            
            # Only keep columns that exist in the dataframe
            all_columns = [col for col in all_columns if col in results_df.columns]
            
            # Reorder columns
            results_df = results_df[all_columns]
            
            # Remove any remaining temporary columns
            temp_columns = ['prev_tx_date', 'tx_date']
            results_df = results_df.drop(columns=[col for col in temp_columns if col in results_df.columns])
            
            # Save to CSV
            results_df.to_csv(
                'anomaly_results/complete_transactions_with_anomalies.csv',
                index=False
            )
            
            # Calculate evaluation metrics for each model
            metrics = {
                'total_transactions': len(self.transactions),
                'anomaly_count': sum(self.transactions['is_anomaly']),
                'anomaly_percentage': (sum(self.transactions['is_anomaly'])/len(self.transactions))*100,
                'threshold_used': float(self.transactions['anomaly_score'].mean() + 
                                      2.0 * self.transactions['anomaly_score'].std())
            }
            
            # Calculate metrics for each anomaly detection model
            for model_name in ['isolationforest', 'ellipticenvelope', 'localoutlierfactor', 'oneclasssvm']:
                model_col = f'anomaly_{model_name}'
                if model_col in self.transactions.columns:
                    y_true = self.transactions['is_anomaly'].astype(int)
                    y_pred = self.transactions[model_col].astype(int)
                    
                    # Calculate metrics with manual verification
                    metrics[f'{model_name}_precision'] = precision_score(y_true, y_pred)
                    metrics[f'{model_name}_recall'] = recall_score(y_true, y_pred)
                    metrics[f'{model_name}_f1'] = f1_score(y_true, y_pred)
                    
                    # Add manual verification metrics
                    if model_name == 'isolationforest':
                        # Verify amount-based anomalies
                        amount_anomalies = self.transactions[
                            (self.transactions[model_col] == 1) & 
                            (self.transactions['amount'] > self.transactions['amount'].quantile(0.95))
                        ]
                        metrics[f'{model_name}_verified_amount_anomalies'] = len(amount_anomalies)
                    
                    elif model_name == 'ellipticenvelope':
                        # Verify tax-related anomalies
                        tax_anomalies = self.transactions[
                            (self.transactions[model_col] == 1) & 
                            (abs(self.transactions['tax_discrepancy']) > self.transactions['tax_discrepancy'].quantile(0.95))
                        ]
                        metrics[f'{model_name}_verified_tax_anomalies'] = len(tax_anomalies)
                    
                    elif model_name == 'localoutlierfactor':
                        # Verify temporal anomalies
                        temporal_anomalies = self.transactions[
                            (self.transactions[model_col] == 1) & 
                            (self.transactions['time_since_last_tx'] < self.transactions['time_since_last_tx'].quantile(0.05))
                        ]
                        metrics[f'{model_name}_verified_temporal_anomalies'] = len(temporal_anomalies)
            
            # Save metrics to CSV
            pd.DataFrame([metrics]).to_csv(
                'evaluation_results/anomaly_detection_metrics.csv',
                index=False
            )
            
            print(f"\nResults saved successfully:")
            print(f"Total transactions: {len(self.transactions)}")
            print(f"Anomalies detected: {metrics['anomaly_count']} ({metrics['anomaly_percentage']:.2f}%)")
            
            # Print model performance metrics
            print("\nModel Performance Metrics:")
            for model_name in ['isolationforest', 'ellipticenvelope', 'localoutlierfactor', 'oneclasssvm']:
                if f'{model_name}_precision' in metrics:
                    print(f"\n{model_name.upper()}:")
                    print(f"Precision: {metrics[f'{model_name}_precision']:.4f}")
                    print(f"Recall: {metrics[f'{model_name}_recall']:.4f}")
                    print(f"F1 Score: {metrics[f'{model_name}_f1']:.4f}")
            
            # Print sample of anomalies
            if metrics['anomaly_count'] > 0:
                print("\nSample anomalous transactions:")
                display_columns = [
                    'transaction_id', 'date', 'amount', 'tax_discrepancy',
                    'is_anomaly', 'anomaly_score'
                ]
                display_columns = [col for col in display_columns if col in results_df.columns]
                print(results_df[results_df['is_anomaly']][display_columns].head().to_string())
            
        except Exception as e:
            print(f"Warning: Could not save results: {str(e)}")
            try:
                results_df.to_csv('complete_transactions_with_anomalies.csv', index=False)
                print("Results saved to current directory instead")
            except Exception as e2:
                print(f"Failed to save results to alternate location: {str(e2)}")

def main():
    """Main execution function"""
    try:
        detector = TransactionAnomalyDetector()
        results = detector.detect_anomalies()
        print("\nAnomaly detection completed successfully.")
    except Exception as e:
        print(f"Error during anomaly detection: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()