import pandas as pd
import os

# Check what CSV files exist and their structure
csv_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

print("Found CSV files:")
for csv_file in csv_files:
    print(f"  {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"    Columns: {list(df.columns)}")
        print(f"    Shape: {df.shape}")
        print(f"    First few rows:")
        print(df.head())
        print()
    except Exception as e:
        print(f"    Error reading file: {e}")
        print()