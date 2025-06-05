import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import io
import base64
import os

def read_dataset(file_path):
    """Read CSV file and return DataFrame"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
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