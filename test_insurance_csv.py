#!/usr/bin/env python3

import os
import sys
import django

# Add the project directory to Python path
sys.path.append('gradient_descent_app')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gradient_descent_app.settings')
django.setup()

from gradient_descent_app.core.utils import read_dataset

def test_insurance_csv():
    """Test reading the problematic insurance CSV file"""
    
    csv_path = 'gradient_descent_app/media/uploads/insuranceCC1.csv'
    
    print("=" * 60)
    print("TESTING INSURANCE CSV FILE PROCESSING")
    print("=" * 60)
    
    try:
        print(f"Testing file: {csv_path}")
        
        # Test reading the CSV
        df = read_dataset(csv_path)
        
        print("\n" + "=" * 40)
        print("SUCCESS! CSV processed successfully")
        print("=" * 40)
        
        print(f"Final dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        print(f"\nLast 5 rows:")
        print(df.tail())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print(f"\nMissing values per column:")
        print(missing_values[missing_values > 0])
        
        # Check target column statistics
        target_col = df.columns[-1]
        print(f"\nTarget column '{target_col}' statistics:")
        print(f"  Min: {df[target_col].min()}")
        print(f"  Max: {df[target_col].max()}")
        print(f"  Mean: {df[target_col].mean():.2f}")
        print(f"  Non-null count: {df[target_col].count()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_insurance_csv()
    if success:
        print("\n✅ Insurance CSV test PASSED!")
    else:
        print("\n❌ Insurance CSV test FAILED!")
        sys.exit(1)