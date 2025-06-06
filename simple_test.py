import pandas as pd
import numpy as np

# Test the problematic test_dataset.csv directly
file_path = "gradient_descent_app/uploads/test_dataset.csv"

print("Testing the problematic CSV file...")
try:
    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Sample:\n{df.head()}")
    
    # Test the smart conversion logic
    feature_cols = df.columns[:-1]
    target_col = df.columns[-1]
    
    print(f"\nTesting smart conversion...")
    numeric_feature_cols = []
    categorical_feature_cols = []
    
    for col in feature_cols:
        test_conversion = pd.to_numeric(df[col], errors='coerce')
        valid_numeric_ratio = test_conversion.notna().sum() / len(df)
        print(f"Column '{col}': {valid_numeric_ratio:.2%} convertible to numeric")
        
        if valid_numeric_ratio >= 0.8:
            numeric_feature_cols.append(col)
        else:
            categorical_feature_cols.append(col)
    
    print(f"Numeric columns: {numeric_feature_cols}")
    print(f"Categorical columns: {categorical_feature_cols}")
    
    if categorical_feature_cols:
        print("Would apply one-hot encoding to categorical columns")
        dummies = pd.get_dummies(df[categorical_feature_cols], prefix=categorical_feature_cols, drop_first=True)
        print(f"One-hot encoded columns: {list(dummies.columns)}")
    
    print("✅ Test successful - the logic would work!")
    
except Exception as e:
    print(f"❌ Error: {e}")