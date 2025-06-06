#!/usr/bin/env python3

import sys
import os
sys.path.append('gradient_descent_app')

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gradient_descent_app.settings')

import django
django.setup()

from gradient_descent_app.core.utils import read_dataset

def test_csv_file(file_path):
    print(f"\n{'='*60}")
    print(f"Testing: {file_path}")
    print(f"{'='*60}")
    
    try:
        df = read_dataset(file_path)
        print(f"✅ SUCCESS! File processed successfully.")
        print(f"Final shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"Sample data:\n{df.head()}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the problematic files
    test_files = [
        "gradient_descent_app/uploads/test_dataset.csv",
        "sample_data.csv"
    ]
    
    success_count = 0
    total_count = 0
    
    for file_path in test_files:
        if os.path.exists(file_path):
            total_count += 1
            if test_csv_file(file_path):
                success_count += 1
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{total_count} files processed successfully")
    print(f"{'='*60}")