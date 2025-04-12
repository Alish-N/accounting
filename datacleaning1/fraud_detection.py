import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import traceback

class FraudDetector:
    def __init__(self):
        self.transactions_file = 'anomaly_results/complete_transactions_with_anomalies.csv'
        self.users_file = 'cleaned_data/cleaned_users.csv'
        self.models = {}
        self.preprocessor = None
        self.feature_names = None
        
        # Update features to only include what's available in the data
        self.numeric_features = [
            'amount',
            'total_tax',
            'tax_discrepancy',
            'amount_zscore',
            'time_since_last_tx',
            'tx_velocity',
            'user_experience_days',
            'amount_per_tax',
            'user_avg_amount',
            'user_std_amount'
        ]
        
        self.categorical_features = [
            'payment_method',
            'transaction_type',
            'category',
            'account_type',
            'currency',
            'status',
            'is_weekend',
            'is_night_hours'
        ]

    def load_and_preprocess_data(self):
        """Load and preprocess transaction and user data"""
        try:
            # Load datasets
            transactions = pd.read_csv(self.transactions_file)
            users = pd.read_csv(self.users_file)
            
            # Convert date columns to datetime with proper timezone handling
            transactions['date'] = pd.to_datetime(transactions['date'], format='mixed').dt.tz_localize(None)
            users['joining_date'] = pd.to_datetime(users['joining_date'], format='mixed').dt.tz_localize(None)
            if 'last_login' in users.columns:
                # Parse ISO8601 format with timezone and convert to timezone naive
                users['last_login'] = pd.to_datetime(users['last_login'], format='mixed')
                if users['last_login'].dt.tz is not None:
                    users['last_login'] = users['last_login'].dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Merge transaction and user data
            merged_data = transactions.merge(
                users,
                left_on='created_by',
                right_on='user_id',
                how='left'
            )
            
            print("\nData shapes:")
            print(f"Transactions: {transactions.shape}")
            print(f"Users: {users.shape}")
            print(f"Merged data: {merged_data.shape}")
            
            # Engineer features
            return self.engineer_features(merged_data)
            
        except Exception as e:
            print(f"\nError in data preprocessing: {str(e)}")
            print("\nDataset shapes:")
            print(f"Transactions: {transactions.shape if 'transactions' in locals() else 'not loaded'}")
            print(f"Users: {users.shape if 'users' in locals() else 'not loaded'}")
            raise

    def engineer_features(self, data, for_prediction=False):
        """Create features for fraud detection using available data"""
        try:
            data = data.copy()
            
            # Basic date-time features (ensure timezone naive)
            data['transaction_date'] = pd.to_datetime(data['date'], format='mixed').dt.tz_localize(None)
            data['hour'] = data['transaction_date'].dt.hour
            data['is_weekend'] = data['transaction_date'].dt.dayofweek.isin([5, 6]).astype(int)
            data['is_night_hours'] = data['hour'].isin([23, 0, 1, 2, 3, 4]).astype(int)
            
            # Amount per tax ratio
            data['amount_per_tax'] = data['amount'] / (data['total_tax'] + 1)
            
            # Transaction velocity features
            data = data.sort_values(['created_by', 'transaction_date'])
            data['time_since_last_tx'] = data.groupby('created_by')['transaction_date'].diff().dt.total_seconds() / 3600
            data['time_since_last_tx'] = data['time_since_last_tx'].fillna(0)
            data['tx_velocity'] = 1 / (data['time_since_last_tx'] + 1)
            
            # Create fraud labels based on available features
            if not for_prediction:
                data['is_fraud'] = (
                    (data['anomaly_score'] > 0.7) &  # High anomaly score
                    (
                        (data['amount_zscore'].abs() > 2.5) |  # Unusual amount
                        (data['tax_discrepancy'] / data['amount'].clip(lower=1) > 0.2) |  # High tax discrepancy
                        (data['tx_velocity'] > data['tx_velocity'].quantile(0.95))  # Unusual transaction velocity
                    )
                ).astype(int)
                
                if 'status' in data.columns:
                    data['is_fraud'] = data['is_fraud'] | (data['status'] == 'Rejected')
            
            # Fill missing values for numeric features
            for feature in self.numeric_features:
                if feature in data.columns:
                    data[feature] = data[feature].fillna(data[feature].mean())
            
            # Select features that exist in the data
            numeric_features = [f for f in self.numeric_features if f in data.columns]
            categorical_features = [f for f in self.categorical_features if f in data.columns]
            
            # Create preprocessor during training
            if not for_prediction and self.preprocessor is None:
                print("\nCreating and fitting preprocessor...")
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
                ])
                
                self.preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ],
                    remainder='drop'
                )
            
            # Prepare feature matrix
            X = data[numeric_features + categorical_features]
            y = data['is_fraud'] if 'is_fraud' in data.columns else None
            
            return X, y
            
        except Exception as e:
            print(f"\nError in feature engineering: {str(e)}")
            print("\nAvailable columns:", data.columns.tolist())
            raise

    def compare_models(self):
        """Train and compare different models for fraud detection"""
        try:
            # Load and preprocess data
            X, y = self.load_and_preprocess_data()
            
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Print class distribution
            print("\nClass Distribution:")
            print(f"Training set - Fraud: {y_train.sum()} ({y_train.mean():.2%})")
            print(f"Test set - Fraud: {y_test.sum()} ({y_test.mean():.2%})")
            
            # Fit the preprocessor on the training data
            print("\nFitting preprocessor on training data...")
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            
            # Store feature names
            try:
                self.feature_names = (
                    self.numeric_features + 
                    [f"{feat}_{val}" for feat, vals in 
                     zip(self.categorical_features, 
                         self.preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_) 
                     for val in vals[1:]]
                )
            except Exception as e:
                print(f"Warning: Could not get feature names: {str(e)}")
            
            # Define models with realistic parameters
            models = {
                'XGBoost': xgb.XGBClassifier(
                    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                    learning_rate=0.005,
                    n_estimators=200,
                    max_depth=4,
                    min_child_weight=5,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    reg_alpha=2,
                    reg_lambda=2,
                    random_state=42
                ),
                'LightGBM': lgb.LGBMClassifier(
                    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                    learning_rate=0.005,
                    n_estimators=200,
                    max_depth=4,
                    num_leaves=16,
                    subsample=0.6,
                    colsample_bytree=0.6,
                    reg_alpha=2,
                    reg_lambda=2,
                    random_state=42
                ),
                'RandomForest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    max_features='sqrt',
                    class_weight='balanced_subsample',
                    random_state=42,
                    n_jobs=-1
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    learning_rate=0.005,
                    n_estimators=200,
                    max_depth=4,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=0.6,
                    max_features='sqrt',
                    random_state=42
                )
            }
            
            # Initialize cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            best_auc = 0
            best_model = None
            best_threshold = None
            
            for name, model in models.items():
                print(f"\nTraining {name}...")
                
                # Perform cross-validation
                cv_scores = []
                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_processed, y_train), 1):
                    # Split data
                    X_fold_train = X_train_processed[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = X_train_processed[val_idx]
                    y_fold_val = y_train.iloc[val_idx]
                    
                    # Apply SMOTE only on training fold
                    if len(y_fold_train[y_fold_train==1]) > 5:
                        smote = SMOTE(
                            sampling_strategy=0.1,  # Create minority class as 10% of majority
                            k_neighbors=min(5, len(y_fold_train[y_fold_train==1]) - 1),
                            random_state=42
                        )
                        X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(
                            X_fold_train, y_fold_train
                        )
                    else:
                        X_fold_train_resampled, y_fold_train_resampled = X_fold_train, y_fold_train
                    
                    # Train model
                    model.fit(X_fold_train_resampled, y_fold_train_resampled)
                    
                    # Evaluate on validation fold
                    y_fold_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                    fold_auc = roc_auc_score(y_fold_val, y_fold_pred_proba)
                    cv_scores.append(fold_auc)
                    
                    print(f"Fold {fold} AUC-ROC: {fold_auc:.4f}")
                
                print(f"\nCross-validation AUC-ROC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
                
                # Retrain on full training set
                if len(y_train[y_train==1]) > 5:
                    smote = SMOTE(
                        sampling_strategy=0.1,
                        k_neighbors=min(5, len(y_train[y_train==1]) - 1),
                        random_state=42
                    )
                    X_train_resampled, y_train_resampled = smote.fit_resample(
                        X_train_processed, y_train
                    )
                else:
                    X_train_resampled, y_train_resampled = X_train_processed, y_train
                
                model.fit(X_train_resampled, y_train_resampled)
                
                # Evaluate on test set
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
                test_auc = roc_auc_score(y_test, y_pred_proba)
                
                print(f"\nTest Set AUC-ROC: {test_auc:.4f}")
                
                # Find optimal threshold using precision-recall curve
                precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]
                
                # Get predictions using optimal threshold
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                
                # Print detailed metrics
                print(f"\nOptimal threshold: {optimal_threshold:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))
                
                print("\nConfusion Matrix:")
                conf_matrix = confusion_matrix(y_test, y_pred)
                print(conf_matrix)
                
                # Calculate additional metrics
                tn, fp, fn, tp = conf_matrix.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                print("\nAdditional Metrics:")
                print(f"Precision (PPV): {precision:.4f}")
                print(f"Recall (Sensitivity): {recall:.4f}")
                print(f"Specificity: {specificity:.4f}")
                print(f"False Positive Rate: {1 - specificity:.4f}")
                print(f"False Negative Rate: {1 - recall:.4f}")
                
                # Print feature importance
                if hasattr(model, 'feature_importances_'):
                    try:
                        feature_importance = pd.DataFrame({
                            'feature': self.feature_names,
                            'importance': model.feature_importances_
                        })
                        feature_importance = feature_importance.sort_values('importance', ascending=False)
                        
                        print("\nTop 10 Most Important Features:")
                        print(feature_importance.head(10).to_string(index=False))
                    except Exception as e:
                        print(f"Error in printing feature importance: {str(e)}")
                
                # Save model
                self.models[name] = {
                    'model': model,
                    'threshold': optimal_threshold,
                    'cv_score': np.mean(cv_scores),
                    'test_score': test_auc
                }
                
                # Update best model
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_model = name
                    best_threshold = optimal_threshold
            
            print(f"\nBest Model: {best_model}")
            print(f"Best AUC-ROC: {best_auc:.4f}")
            print(f"Best Threshold: {best_threshold:.4f}")
            
            return {
                'best_model': best_model,
                'auc': best_auc,
                'threshold': best_threshold,
                'models': self.models
            }
            
        except Exception as e:
            print(f"\nError during model comparison: {str(e)}")
            traceback.print_exc()
            raise

    def _print_feature_importance(self, model):
        """Print feature importance for a given model"""
        try:
            # Get feature names directly
            feature_names = self.numeric_features + self.categorical_features
            
            # Create feature importance DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
            
        except Exception as e:
            print(f"\nError in printing feature importance: {str(e)}")
            # Continue execution even if feature importance printing fails
            pass

    def predict_fraud(self, transaction_data):
        """Predict fraud probability for new transactions"""
        try:
            # Make a copy to avoid modifying the original
            transaction_data = transaction_data.copy()
            
            # Map column names if needed
            if 'status' in transaction_data.columns and 'status_x' not in transaction_data.columns:
                transaction_data['status_x'] = transaction_data['status']
            
            # Engineer features for prediction data (with for_prediction=True)
            X, _ = self.engineer_features(transaction_data, for_prediction=True)
            
            if self.preprocessor is None:
                raise ValueError("Preprocessor not fitted. Train models first.")
                
            print(f"Transforming data with preprocessor...")
            
            # Get predictions from all models
            predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    # Transform the data using the fitted preprocessor
                    X_processed = self.preprocessor.transform(X)
                    
                    # Make predictions
                    y_pred_proba = model['model'].predict_proba(X_processed)[:, 1]
                    predictions[model_name] = y_pred_proba
                    print(f"Successfully made predictions with {model_name}")
                except Exception as e:
                    print(f"Warning: Error in {model_name} prediction: {str(e)}")
                    continue
            
            if not predictions:
                raise ValueError("No models were able to make predictions")
            
            # Calculate ensemble prediction
            ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
            
            # Create results DataFrame
            results_dict = {
                'transaction_id': transaction_data['transaction_id'],
                'date': transaction_data['date'],
                'amount': transaction_data['amount'],
                'payment_method': transaction_data['payment_method'],
                'category': transaction_data['category'],
                'anomaly_score': transaction_data['anomaly_score'],
                'fraud_probability': ensemble_pred,
                'is_fraud': ensemble_pred > self.models[model_name]['threshold'],
                'risk_level': pd.cut(ensemble_pred, 
                               bins=[0, 0.3, 0.5, 0.7, 1.0],
                               labels=['Low', 'Medium', 'High', 'Critical'])
            }
            
            # Add status if it exists
            if 'status' in transaction_data.columns:
                results_dict['status'] = transaction_data['status']
            
            results = pd.DataFrame(results_dict)
            
            return results
            
        except Exception as e:
            print(f"Error in fraud prediction: {str(e)}")
            print("\nAvailable columns:", transaction_data.columns.tolist())
            raise

    def generate_fraud_report(self, predictions, output_file='fraud_results/fraud_report.csv'):
        """Generate a report of fraud predictions"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save predictions to CSV
            predictions.to_csv(output_file, index=False)
            
            # Print summary
            print("\nFraud Detection Summary:")
            print(f"Total transactions analyzed: {len(predictions)}")
            print(f"Fraudulent transactions detected: {predictions['is_fraud'].sum()}")
            print(f"Fraud rate: {predictions['is_fraud'].sum() / len(predictions):.2%}")
            
            # Print risk level distribution
            risk_counts = predictions['risk_level'].value_counts()
            print("\nRisk Level Distribution:")
            print("=" * 50)
            print(f"{'Risk Level':<15} {'Count':<10} {'Percentage':<15} {'Avg Probability':<15}")
            print("-" * 50)
            
            for level in ['Low', 'Medium', 'High', 'Critical']:
                if level in risk_counts:
                    count = risk_counts[level]
                    percentage = count / len(predictions) * 100
                    avg_prob = predictions[predictions['risk_level'] == level]['fraud_probability'].mean()
                    print(f"{level:<15} {count:<10} {percentage:>6.2f}%        {avg_prob:.4f}")
            
            print("=" * 50)
            
            # Print top 10 highest risk transactions
            print("\nTop 10 Highest Risk Transactions:")
            print("=" * 80)
            high_risk = predictions.sort_values('fraud_probability', ascending=False).head(10)
            
            # Format as a table
            print(f"{'Transaction ID':<15} {'Amount':<10} {'Category':<20} {'Fraud Prob':<12} {'Risk Level':<10}")
            print("-" * 80)
            
            for _, row in high_risk.iterrows():
                print(f"{row['transaction_id']:<15} {row['amount']:<10.2f} {row['category']:<20} {row['fraud_probability']:.4f}      {row['risk_level']}")
            
            print("=" * 80)
            
            # Print fraud by category
            print("\nFraud Distribution by Category:")
            print("=" * 60)
            category_fraud = predictions.groupby('category')['is_fraud'].agg(['count', 'sum'])
            category_fraud['rate'] = category_fraud['sum'] / category_fraud['count'] * 100
            category_fraud = category_fraud.sort_values('rate', ascending=False)
            
            print(f"{'Category':<25} {'Total':<8} {'Fraud':<8} {'Rate':<8}")
            print("-" * 60)
            
            for category, row in category_fraud.iterrows():
                if row['sum'] > 0:
                    print(f"{category:<25} {row['count']:<8} {row['sum']:<8} {row['rate']:>6.2f}%")
            
            print("=" * 60)
            
            print(f"\nFraud report saved to {output_file}")
            
            return True
            
        except Exception as e:
            print(f"Error generating fraud report: {str(e)}")
            return False

def main():
    """Main function to run the fraud detection system"""
    try:
        print("\nInitializing Fraud Detection System...")
        detector = FraudDetector()
        
        print("\nTraining models...")
        detector.compare_models()
        
        print("\nLoading transaction data for prediction...")
        transactions = pd.read_csv('anomaly_results/complete_transactions_with_anomalies.csv')
        
        print("\nPredicting fraud...")
        predictions = detector.predict_fraud(transactions)
        
        print("\nGenerating fraud report...")
        detector.generate_fraud_report(predictions)
        
        print("\nFraud detection completed successfully!")
        
    except Exception as e:
        print(f"Error during fraud detection: {str(e)}")
        traceback.print_exc()
        
if __name__ == "__main__":
    main()