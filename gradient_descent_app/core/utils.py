import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import io
import base64
import os

def read_dataset(file_path):
    """Read CSV file and return DataFrame with intelligent handling of mixed data types"""
    try:
        print(f"[DEBUG] Reading CSV file: {file_path}")
        
        # Try different encoding and delimiter combinations
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        separators = [None, ',', ';', '\t']  # None means auto-detect
        
        df = None
        successful_config = None
        
        for encoding in encodings:
            for sep in separators:
                try:
                    if sep is None:
                        # Auto-detect separator using python engine
                        df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
                    else:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                    
                    # Basic validation of the read result
                    if df is not None and not df.empty and len(df.columns) >= 2:
                        successful_config = f"encoding={encoding}, sep={sep if sep else 'auto-detect'}"
                        print(f"[DEBUG] Successfully read with {successful_config}")
                        break
                except Exception as read_error:
                    print(f"[DEBUG] Failed with encoding={encoding}, sep={sep}: {read_error}")
                    continue
            
            if df is not None and not df.empty and len(df.columns) >= 2:
                break
        
        if df is None or df.empty:
            raise Exception("Could not read CSV file with any encoding/delimiter combination")
        
        # Clean column names (remove whitespace and handle unnamed columns)
        df.rename(columns=lambda x: x.strip(), inplace=True)
        
        # Remove completely empty columns or columns with only NaN
        df = df.dropna(axis=1, how='all')
        
        # Remove unnamed/empty columns that might be artifacts
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        print(f"[DEBUG] Cleaned column names: {list(df.columns)}")
        
        # Validate minimum columns after cleaning
        if len(df.columns) < 2:
            raise Exception(f"CSV file must have at least 2 valid columns (features + target), found {len(df.columns)} after cleaning")
        
        # Log original data types and sample
        print(f"[DEBUG] Original data types:\n{df.dtypes}")
        print(f"[DEBUG] Original data sample:\n{df.head()}")
        print(f"[DEBUG] Original shape: {df.shape}")
        
        # Identify which columns can be converted to numeric
        feature_cols = df.columns[:-1]
        target_col = df.columns[-1]
        
        print(f"[DEBUG] Analyzing {len(feature_cols)} feature columns for numeric conversion...")
        
        # Smart conversion: only convert columns that have mostly numeric data
        numeric_feature_cols = []
        categorical_feature_cols = []
        
        for col in feature_cols:
            # Try converting to see how many values are convertible
            test_conversion = pd.to_numeric(df[col], errors='coerce')
            valid_numeric_ratio = test_conversion.notna().sum() / len(df)
            
            print(f"[DEBUG] Column '{col}': {test_conversion.notna().sum()}/{len(df)} values convertible to numeric ({valid_numeric_ratio:.2%})")
            
            if valid_numeric_ratio >= 0.8:  # If 80% or more can be converted, treat as numeric
                numeric_feature_cols.append(col)
                df[col] = test_conversion
            else:
                categorical_feature_cols.append(col)
                print(f"[DEBUG] Column '{col}' treated as categorical (too few numeric values)")
        
        # Convert target column to numeric (required)
        print(f"[DEBUG] Converting target column '{target_col}' to numeric...")
        original_target_non_null = df[target_col].notna().sum()
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        converted_target_non_null = df[target_col].notna().sum()
        print(f"[DEBUG] Target column '{target_col}': {original_target_non_null} -> {converted_target_non_null} valid values")
        
        # Remove rows with missing target values first
        initial_rows = len(df)
        df = df.dropna(subset=[target_col])
        dropped_target_rows = initial_rows - len(df)
        print(f"[DEBUG] Dropped {dropped_target_rows} rows with missing target values. Remaining: {len(df)}")
        
        if len(df) == 0:
            raise Exception(f"Target column '{target_col}' contains no valid numeric values after removing missing data. Please ensure the last column contains numeric target values.")
        
        # Handle categorical columns using one-hot encoding
        if categorical_feature_cols:
            print(f"[DEBUG] Applying one-hot encoding to categorical columns: {categorical_feature_cols}")
            
            # Fill missing values in categorical columns before one-hot encoding
            for col in categorical_feature_cols:
                if df[col].isnull().any():
                    missing_count = df[col].isnull().sum()
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        fill_val = mode_val[0]
                    else:
                        fill_val = 'Unknown'
                    df[col].fillna(fill_val, inplace=True)
                    print(f"[DEBUG] Filled {missing_count} missing values in categorical '{col}' with: {fill_val}")
            
            # Create dummy variables for categorical columns
            categorical_dummies = pd.get_dummies(df[categorical_feature_cols], prefix=categorical_feature_cols, drop_first=True)
            
            # Remove original categorical columns and add dummy columns
            df = df.drop(columns=categorical_feature_cols)
            df = pd.concat([df.iloc[:, :-1], categorical_dummies, df.iloc[:, -1:]], axis=1)
            
            print(f"[DEBUG] After one-hot encoding, new columns: {list(df.columns)}")
        
        # Log data types after conversion
        print(f"[DEBUG] Data types after processing:\n{df.dtypes}")
        
        # Check if we have any numeric feature columns
        feature_columns = df.columns[:-1]  # All except target
        numeric_features = df[feature_columns].select_dtypes(include=['float64', 'int64'])
        
        if numeric_features.empty:
            raise Exception("No numeric feature columns found after processing. Please ensure your CSV contains numeric data for features.")
        
        print(f"[DEBUG] Found {len(numeric_features.columns)} numeric feature columns: {list(numeric_features.columns)}")
        
        # Only drop rows where the target is NaN (be more lenient with features)
        rows_before = len(df)
        df = df.dropna(subset=[target_col])  # Only require target to be non-null
        rows_after = len(df)
        print(f"[DEBUG] Dropped {rows_before - rows_after} rows with missing target values. Remaining: {rows_after}")
        
        # For remaining NaN values in features, fill with median/mode
        for col in df.columns[:-1]:  # All feature columns
            if df[col].dtype in ['float64', 'int64']:
                # Fill numeric columns with median
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"[DEBUG] Filled {col} NaN values with median: {median_val}")
        
        if df.empty:
            raise Exception("No valid rows remaining after processing. Please check your CSV file format.")
        
        # Final validation
        if len(df) < 5:
            raise Exception(f"Insufficient data: only {len(df)} valid rows found. Need at least 5 rows for meaningful analysis.")
        
        # Ensure we have at least one numeric feature
        final_numeric_features = df.iloc[:, :-1].select_dtypes(include=['float64', 'int64'])
        if final_numeric_features.empty:
            raise Exception("No numeric features available for analysis after processing.")
        
        # Log final sample
        print(f"[DEBUG] Final data sample:\n{df.head()}")
        print(f"[DEBUG] Final shape: {df.shape}")
        print(f"[DEBUG] Final feature columns: {list(df.columns[:-1])}")
        print(f"[DEBUG] Target column: {df.columns[-1]}")
        print(f"[DEBUG] Successfully processed CSV file with {successful_config}")
        
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {str(e)}")
        raise Exception(f"Error reading CSV file: {str(e)}")

def get_descriptive_stats(df):
    """Get descriptive statistics for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {}
    
    stats = {}
    for col in numeric_df.columns:
        stats[col] = {
            'mean': numeric_df[col].mean(),
            'median': numeric_df[col].median(),
            'std': numeric_df[col].std(),
            'min': numeric_df[col].min(),
            'max': numeric_df[col].max(),
            'null_count': df[col].isnull().sum()
        }
    return stats

def handle_missing_values(df, method, fill_value=None):
    """Handle missing values in DataFrame"""
    df_copy = df.copy()
    
    if method == 'drop':
        df_copy = df_copy.dropna()
    elif method == 'mean':
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
    elif method == 'constant' and fill_value is not None:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(fill_value)
    
    return df_copy

def scale_features(df, method):
    """Scale numeric features"""
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return df_copy
    
    if method == 'standard':
        scaler = StandardScaler()
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    elif method == 'minmax':
        scaler = MinMaxScaler()
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    
    return df_copy

def encode_categorical(df):
    """Encode categorical variables using one-hot encoding"""
    return pd.get_dummies(df, drop_first=True)

def split_data(df, target_col, test_size=0.2):
    """Split data into train and test sets"""
    if target_col not in df.columns:
        # Use last column as target if target_col not found
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def add_bias_column(X):
    """Add bias column (column of ones) to feature matrix"""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def gradient_descent_linear(X, y, alpha, num_iters):
    """Implement gradient descent for linear regression"""
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    
    for i in range(num_iters):
        # Forward propagation
        predictions = X.dot(theta)
        errors = predictions - y
        
        # Compute cost
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)
        
        # Compute gradients
        gradients = (1 / m) * X.T.dot(errors)
        
        # Update parameters
        theta -= alpha * gradients
    
    return theta, cost_history

def sigmoid(z):
    """Sigmoid activation function"""
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def gradient_descent_logistic(X, y, alpha, num_iters):
    """Implement gradient descent for logistic regression"""
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    
    for i in range(num_iters):
        # Forward propagation
        z = X.dot(theta)
        predictions = sigmoid(z)
        
        # Compute cost (log-likelihood)
        epsilon = 1e-15  # Small value to prevent log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        cost = -(1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        cost_history.append(cost)
        
        # Compute gradients
        errors = predictions - y
        gradients = (1 / m) * X.T.dot(errors)
        
        # Update parameters
        theta -= alpha * gradients
    
    return theta, cost_history

def plot_learning_curve(cost_history):
    """Plot learning curve and return base64 encoded image"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(cost_history)), cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Learning Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

def validate_csv_file(file):
    """Validate uploaded CSV file"""
    # Check file extension
    if not file.name.endswith('.csv'):
        return False, "Le fichier doit être de type CSV"
    
    # Check file size (5MB limit)
    if file.size > 5 * 1024 * 1024:
        return False, "La taille du fichier doit être inférieure à 5 MB"
    
    return True, "Fichier valide"

def ensure_media_dirs():
    """Ensure media directories exist"""
    from django.conf import settings
    
    uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
    
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    return uploads_dir, processed_dir