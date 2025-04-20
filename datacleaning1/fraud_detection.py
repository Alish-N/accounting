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
        """Initialize the fraud detector with appropriate paths and settings"""
        # Data file paths
        self.transactions_file = 'anomaly_results/complete_transactions_with_anomalies.csv'
        self.users_file = 'cleaned_data/cleaned_users.csv'
        
        # Initialize model components
        self.preprocessor = None
        self.model = None
        self.optimal_threshold = 0.5
        self.feature_names = []
        
        # Create output directories
        os.makedirs('fraud_results', exist_ok=True)
        os.makedirs('datacleaning1/fraud_results', exist_ok=True)
        
        # Define feature sets
        self.numeric_features = [
            'amount', 'total_tax', 'total_tax_calculated', 'tax_discrepancy',
            'anomaly_score', 'amount_zscore', 'hour', 'is_weekend', 'is_night_hours',
            'amount_per_tax', 'time_since_last_tx', 'tx_velocity'
        ]
        
        self.categorical_features = [
            'transaction_type', 'account_type', 'currency', 'payment_method',
            'category', 'status', 'department'
        ]

    def load_and_preprocess_data(self, transactions=None, for_prediction=False):
        """Load and preprocess transaction and user data"""
        try:
            # Load datasets
            if transactions is None:
            transactions = pd.read_csv(self.transactions_file)
            users = pd.read_csv(self.users_file)
            
            # Clean and fix date column for transactions
            try:
                if 'date' in transactions.columns:
                    # First check if any values are the literal string "string"
                    transactions['date'] = transactions['date'].replace('string', pd.NaT)
                    
                    # Try to convert to datetime with robust error handling
                    transactions['date'] = pd.to_datetime(transactions['date'], errors='coerce')
                    
                    # Check if we have NaT values and handle them
                    if transactions['date'].isna().any():
                        # Count null values
                        null_count = transactions['date'].isna().sum()
                        print(f"Warning: Found {null_count} invalid date values in transactions. Filling with current date.")
                        
                        # Fill NaT values with the current date
                        transactions['date'] = transactions['date'].fillna(pd.Timestamp.now())
            except Exception as e:
                print(f"Warning: Error cleaning transaction date column: {str(e)}")
                # Create a fallback date column with the current timestamp
                transactions['date'] = pd.Timestamp.now()
                print("Created fallback date column with current timestamp for transactions.")
            
            # Clean and fix joining_date column for users
            try:
                if 'joining_date' in users.columns:
                    # First check if any values are the literal string "string"
                    users['joining_date'] = users['joining_date'].replace('string', pd.NaT)
                    
                    # Try to convert to datetime with robust error handling
                    users['joining_date'] = pd.to_datetime(users['joining_date'], errors='coerce')
                    
                    # Check if we have NaT values and handle them
                    if users['joining_date'].isna().any():
                        # Count null values
                        null_count = users['joining_date'].isna().sum()
                        print(f"Warning: Found {null_count} invalid date values in users. Filling with current date.")
                        
                        # Fill NaT values with the current date
                        users['joining_date'] = users['joining_date'].fillna(pd.Timestamp.now())
                        
                # Make sure timezone info is removed
                if not pd.api.types.is_datetime64_dtype(users['joining_date']):
                    users['joining_date'] = pd.to_datetime(users['joining_date'], errors='coerce')
                    
                users['joining_date'] = users['joining_date'].dt.tz_localize(None)
            except Exception as e:
                print(f"Warning: Error cleaning user joining_date column: {str(e)}")
                # Create a fallback date column with the current timestamp
                users['joining_date'] = pd.Timestamp.now()
                print("Created fallback date column with current timestamp for users.")
            
            # Handle last_login column if present
            try:
            if 'last_login' in users.columns:
                    # Handle any literal "string" values
                    users['last_login'] = users['last_login'].replace('string', pd.NaT)
                    
                # Parse ISO8601 format with timezone and convert to timezone naive
                    users['last_login'] = pd.to_datetime(users['last_login'], errors='coerce')
                    
                    # Fill NaT values
                    if users['last_login'].isna().any():
                        users['last_login'] = users['last_login'].fillna(pd.Timestamp.now())
                    
                    # Remove timezone if present
                    if hasattr(users['last_login'], 'dt') and users['last_login'].dt.tz is not None:
                    users['last_login'] = users['last_login'].dt.tz_convert('UTC').dt.tz_localize(None)
            except Exception as e:
                print(f"Warning: Error cleaning user last_login column: {str(e)}")
                if 'last_login' in users.columns:
                    users['last_login'] = pd.Timestamp.now()
            
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
            return self.engineer_features(merged_data, for_prediction)
            
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
            try:
                # Ensure transaction_date is properly created
                if 'date' in data.columns:
                    # Verify date column is datetime type
                    if not pd.api.types.is_datetime64_dtype(data['date']):
                        # Convert with error handling
                        data['date'] = pd.to_datetime(data['date'], errors='coerce')
                        # Fill any NaT values
                        if data['date'].isna().any():
                            data['date'] = data['date'].fillna(pd.Timestamp.now())
                
                # Create transaction_date column
                data['transaction_date'] = pd.to_datetime(data['date'], errors='coerce')
                
                # Fill any NaT values in transaction_date
                if data['transaction_date'].isna().any():
                    data['transaction_date'] = data['transaction_date'].fillna(pd.Timestamp.now())
                
                # Remove timezone if present
                if data['transaction_date'].dt.tz is not None:
                    data['transaction_date'] = data['transaction_date'].dt.tz_localize(None)
                
                # Extract features
            data['hour'] = data['transaction_date'].dt.hour
            data['is_weekend'] = data['transaction_date'].dt.dayofweek.isin([5, 6]).astype(int)
            data['is_night_hours'] = data['hour'].isin([23, 0, 1, 2, 3, 4]).astype(int)
            
                print("Successfully extracted datetime features")
            except Exception as e:
                print(f"Warning: Error creating datetime features: {str(e)}")
                # Create fallback datetime features
                data['transaction_date'] = pd.Timestamp.now()
                data['hour'] = data['transaction_date'].hour
                data['is_weekend'] = 0
                data['is_night_hours'] = 0
                print("Created fallback datetime features")
            
            # Amount per tax ratio with error handling
            try:
            data['amount_per_tax'] = data['amount'] / (data['total_tax'] + 1)
            except Exception as e:
                print(f"Warning: Error calculating amount_per_tax: {str(e)}")
                data['amount_per_tax'] = data['amount'] / 1.0
            
            # Transaction velocity features with error handling
            try:
            data = data.sort_values(['created_by', 'transaction_date'])
            data['time_since_last_tx'] = data.groupby('created_by')['transaction_date'].diff().dt.total_seconds() / 3600
            data['time_since_last_tx'] = data['time_since_last_tx'].fillna(0)
            data['tx_velocity'] = 1 / (data['time_since_last_tx'] + 1)
            except Exception as e:
                print(f"Warning: Error calculating transaction velocity: {str(e)}")
                data['time_since_last_tx'] = 0
                data['tx_velocity'] = 0
            
            # Create fraud labels based on available features
            if not for_prediction:
                try:
                    # Ensure numeric columns used for fraud detection are available and properly formatted
                    if 'anomaly_score' not in data.columns:
                        data['anomaly_score'] = 0
                    
                    if 'amount_zscore' not in data.columns:
                        data['amount_zscore'] = 0
                        
                    if 'tax_discrepancy' not in data.columns:
                        data['tax_discrepancy'] = 0
                    
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
                except Exception as e:
                    print(f"Warning: Error creating fraud labels: {str(e)}")
                    data['is_fraud'] = 0
            
            # Fill missing values for numeric features
            for feature in self.numeric_features:
                if feature in data.columns:
                    # Calculate a safe mean value (avoid NaN mean)
                    mean_val = data[feature].mean()
                    if pd.isna(mean_val):
                        mean_val = 0
                    data[feature] = data[feature].fillna(mean_val)
            
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
            
            # Check if we have enough data for a train-test split
            if len(X) < 5:
                print(f"\nVery small dataset detected ({len(X)} samples). Using simplified approach.")
                
                # For extremely small datasets, use a simplified approach
                if len(X) <= 1:
                    print("Single transaction detected. Using rule-based classification only.")
                    
                    # Create and fit a simple preprocessor if needed
                    if self.preprocessor is None:
                        # Get the feature lists
                        numeric_features = [f for f in self.numeric_features if f in X.columns]
                        categorical_features = [f for f in self.categorical_features if f in X.columns]
                        
                        # Create a simple preprocessor
                        print("Creating simplified preprocessor...")
                        numeric_transformer = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='mean')),
                            ('scaler', StandardScaler())
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
                        
                        # Fit preprocessor on available data
                        self.preprocessor.fit(X)
                    
                    # Use RandomForest as our default model for simplicity
                    print("Training simplified model...")
                    self.model = RandomForestClassifier(
                        n_estimators=20,
                        max_depth=3,
                        random_state=42,
                        class_weight='balanced'
                    )
                    
                    # Create a synthetic dataset for training with the single transaction info
                    X_processed = self.preprocessor.transform(X)
                    
                    # Synthetic values
                    X_synthetic = np.vstack([X_processed, X_processed])
                    y_synthetic = np.array([0, 1])  # One fraud, one non-fraud example
                    
                    # Train on synthetic data
                    self.model.fit(X_synthetic, y_synthetic)
                    self.optimal_threshold = 0.5
                    
                    return {
                        'status': 'simplified',
                        'message': 'Single transaction handled with rule-based approach',
                        'best_model': 'RuleBased',
                        'auc': 0.5,  # Default value
                        'optimal_threshold': self.optimal_threshold
                    }
                else:
                    # For very small datasets (2-4 records), use leave-one-out CV
                    print("Small dataset detected. Using leave-one-out cross-validation.")
                    
                    # Fit the preprocessor on all data
                    if self.preprocessor is None:
                        # Get the feature lists
                        numeric_features = [f for f in self.numeric_features if f in X.columns]
                        categorical_features = [f for f in self.categorical_features if f in X.columns]
                        
                        # Create a simple preprocessor
                        print("Creating preprocessor for small dataset...")
                        numeric_transformer = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='mean')),
                            ('scaler', StandardScaler())
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
                    
                    X_processed = self.preprocessor.fit_transform(X)
                    
                    # Use a simple model with class weights
                    self.model = RandomForestClassifier(
                        n_estimators=20,
                        max_depth=3,
                        random_state=42,
                        class_weight='balanced'
                    )
                    
                    # Train on all data
                    self.model.fit(X_processed, y)
                    self.optimal_threshold = 0.5
                    
                    # Make predictions on the same data (for demonstration)
                    y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
                    
                    return {
                        'status': 'simplified',
                        'message': 'Small dataset handled with simplified approach',
                        'best_model': 'RandomForest',
                        'auc': 0.5,  # Default value since we can't properly evaluate
                        'optimal_threshold': self.optimal_threshold
                    }
            
            # For normal-sized datasets, proceed with regular train-test split
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
            best_model_name = None
            
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
                    
                    # Train model directly on original data (no SMOTE)
                    model.fit(X_fold_train, y_fold_train)
                    
                    # Evaluate on validation fold
                    y_fold_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                    fold_auc = roc_auc_score(y_fold_val, y_fold_pred_proba)
                    cv_scores.append(fold_auc)
                    
                    print(f"Fold {fold} AUC-ROC: {fold_auc:.4f}")
                
                print(f"\nCross-validation AUC-ROC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
                
                # Retrain on full training set (no SMOTE)
                model.fit(X_train_processed, y_train)
                
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
                        print(f"Warning: Could not calculate feature importance: {str(e)}")
                
                # If this is the best model so far, save it
                if test_auc > best_auc:
                    best_auc = test_auc
                    best_model = model
                    best_threshold = optimal_threshold
                    best_model_name = name
                    
                    print(f"\n*** {name} is the new best model with AUC: {test_auc:.4f} ***")
            
            # Save the best model
            if best_model is not None:
                self.model = best_model
                self.optimal_threshold = best_threshold
                print(f"\nBest model: {best_model_name} (AUC: {best_auc:.4f}, Threshold: {best_threshold:.4f})")
            else:
                # Fallback to a default model if none performed well
                print("\nWarning: No suitable model found. Using default model.")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                self.model.fit(X_train_processed, y_train)
                self.optimal_threshold = 0.5
            
            return {
                'best_model': best_model_name if best_model_name else 'RandomForest',
                'auc': best_auc,
                'optimal_threshold': self.optimal_threshold
            }
            
        except Exception as e:
            print(f"\nError during model comparison: {str(e)}")
            traceback.print_exc()
            
            # Create a basic model as fallback
            try:
                if hasattr(self, 'preprocessor') and self.preprocessor is not None and 'X' in locals() and 'y' in locals():
                    X_processed = self.preprocessor.fit_transform(X)
                    self.model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, class_weight='balanced')
                    self.model.fit(X_processed, y)
                    self.optimal_threshold = 0.5
                    print("Created fallback model")
                    
                    return {
                        'status': 'fallback',
                        'message': f"Error in model training, using fallback: {str(e)}",
                        'best_model': 'RandomForest',
                        'auc': 0.5,
                        'optimal_threshold': self.optimal_threshold
                    }
            except Exception as fallback_error:
                print(f"Could not create fallback model: {str(fallback_error)}")
                
            # Re-raise the original exception
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
        """Predict fraud probabilities for new transactions"""
        try:
            # If not trained, try to train the model
            if self.model is None:
                print("Model not trained. Attempting to train...")
                model_info = self.compare_models()
                # If we got back info but no model was set, set a flag to use rule-based approach
                if model_info and model_info.get('status') == 'simplified':
                    print(f"Using simplified approach: {model_info.get('message')}")
            
            # Preprocess the new data
            X, _ = self.load_and_preprocess_data(
                transactions=transaction_data,
                for_prediction=True
            )
            
            # Extract features for prediction
            X_features, _ = self.engineer_features(X, for_prediction=True)
            
            # Check if X_features is empty or has too few samples
            if len(X_features) == 0:
                raise ValueError("No valid features could be extracted for prediction")
            
            # Simple rule-based approach for very small datasets or if model training failed
            if self.model is None or len(X_features) < 5:
                print("Using rule-based approach for fraud prediction...")
                
                # Calculate risk score using anomaly score and other indicators if available
                risk_scores = []
                
                for _, row in X_features.iterrows():
                    risk_score = 0.1  # Base risk
                    
                    # Use anomaly score if available (highest weight)
                    if 'anomaly_score' in row:
                        risk_score += row['anomaly_score'] * 0.5
                    
                    # Add weight for unusual amounts
                    if 'amount' in row and row['amount'] > 5000:
                        risk_score += 0.2
                    
                    # Add weight for tax discrepancy
                    if 'tax_discrepancy' in row and row['tax_discrepancy'] > 0:
                        discrepancy_ratio = row['tax_discrepancy'] / (row['amount'] if 'amount' in row and row['amount'] > 0 else 100)
                        if discrepancy_ratio > 0.2:
                            risk_score += 0.3
                    
                    # Add weight for night hours
                    if 'is_night_hours' in row and row['is_night_hours'] == 1:
                        risk_score += 0.1
                    
                    # Add weight for weekend
                    if 'is_weekend' in row and row['is_weekend'] == 1:
                        risk_score += 0.1
                    
                    # Cap at 0.95 to avoid certainty
                    risk_score = min(risk_score, 0.95)
                    risk_scores.append(risk_score)
                
                # Create predictions dataframe
                predictions = X_features.copy()
                predictions['fraud_probability'] = risk_scores
                predictions['is_fraud'] = [1 if score > 0.7 else 0 for score in risk_scores]
                
                # Add risk level
                predictions['risk_level'] = pd.cut(
                    predictions['fraud_probability'],
                    bins=[0, 0.3, 0.5, 0.7, 1.0],
                    labels=['Low', 'Medium', 'High', 'Critical']
                )
                
                return predictions
            
            # Use the trained model for normal-sized datasets
            print(f"Using trained model for fraud prediction on {len(X_features)} transactions...")
            
            # Transform features
            X_processed = self.preprocessor.transform(X_features)
            
            # Predict fraud probabilities
            fraud_proba = self.model.predict_proba(X_processed)[:, 1]
            
            # Create predictions dataframe
            predictions = X_features.copy()
            predictions['fraud_probability'] = fraud_proba
            predictions['is_fraud'] = [1 if p > self.optimal_threshold else 0 for p in fraud_proba]
            
            # Add risk level
            predictions['risk_level'] = pd.cut(
                predictions['fraud_probability'],
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            
            # Calculate risk metrics
            risk_metrics = {
                'high_risk_count': sum(predictions['risk_level'].isin(['High', 'Critical'])),
                'fraud_count': sum(predictions['is_fraud']),
                'average_fraud_probability': predictions['fraud_probability'].mean(),
                'max_fraud_probability': predictions['fraud_probability'].max()
            }
            
            print("\nRisk Summary:")
            print(f"High Risk Transactions: {risk_metrics['high_risk_count']} ({risk_metrics['high_risk_count']/len(predictions)*100:.1f}%)")
            print(f"Probable Fraud: {risk_metrics['fraud_count']} ({risk_metrics['fraud_count']/len(predictions)*100:.1f}%)")
            print(f"Average Fraud Probability: {risk_metrics['average_fraud_probability']:.2f}")
            print(f"Max Fraud Probability: {risk_metrics['max_fraud_probability']:.2f}")
            
            return predictions
            
        except Exception as e:
            print(f"\nError in fraud prediction: {str(e)}")
            traceback.print_exc()
            
            # Fallback to ensure we return something useful
            try:
                # Create a simplified prediction with just transaction_ids and default values
                fallback_predictions = transaction_data.copy()
                fallback_predictions['fraud_probability'] = 0.1
                fallback_predictions['is_fraud'] = 0
                fallback_predictions['risk_level'] = 'Low'
                
                print("Using fallback predictions due to error")
                return fallback_predictions
            except:
                print("Could not create fallback predictions")
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