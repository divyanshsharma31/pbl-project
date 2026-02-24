import pandas as pd
import numpy as np
import os

print("="*70)
print("DATASET CLEANING AND FINE-TUNING")
print("="*70)

input_file = 'files/station_hour_transformed.csv'
output_file = 'files/station_hour_cleaned.csv'

print(f"\nReading dataset from: {input_file}")

# Get original file size
original_size = os.path.getsize(input_file) / (1024**2)
print(f"Original file size: {original_size:.2f} MB")

# Read and process the dataset
df = pd.read_csv(input_file)

print(f"\nOriginal dataset shape: {df.shape}")
print(f"Total rows: {len(df):,}")

# Step 1: Remove rows with NaN/Unknown AQI values
print(f"\nRemoving rows with unknown (NaN) AQI values...")
aqi_null_count = df['AQI'].isnull().sum()
print(f"  - Rows with NaN AQI: {aqi_null_count:,}")

df = df[df['AQI'].notna()]
print(f"  - Rows after removing NaN AQI: {len(df):,}")

# Step 2: Remove rows with NaN in critical columns
print(f"\nRemoving rows with NaN in critical columns...")
critical_cols = ['StationId', 'Datetime', 'PM2.5', 'PM10', 'AQI', 'AQI_Bucket', 'StationName', 'City', 'State']
rows_before = len(df)
df = df.dropna(subset=critical_cols)
critical_null_removed = rows_before - len(df)
print(f"  - Rows removed due to null in critical columns: {critical_null_removed:,}")

# Step 3: Drop unnecessary index column
print(f"\nCleaning up columns...")
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)
    print(f"  - Removed 'Unnamed: 0' index column")

# Step 4: Remove duplicate rows (if any)
print(f"\nRemoving duplicates...")
duplicates_before = len(df)
df = df.drop_duplicates()
duplicates_removed = duplicates_before - len(df)
print(f"  - Duplicate rows removed: {duplicates_removed:,}")

# Step 5: Validate AQI values are reasonable
print(f"\nValidating AQI values...")
print(f"  - AQI range: {df['AQI'].min():.1f} to {df['AQI'].max():.1f}")
print(f"  - AQI mean: {df['AQI'].mean():.2f}")
print(f"  - AQI median: {df['AQI'].median():.2f}")

# Step 6: Data quality checks
print(f"\nData quality checks:")
print(f"  - Remaining null values by column:")
null_counts = df.isnull().sum()
nulls = null_counts[null_counts > 0]
if len(nulls) > 0:
    print(nulls)
else:
    print("    None - Data is clean!")

# Step 7: Save the cleaned dataset
print(f"\nSaving cleaned dataset to: {output_file}")
df.to_csv(output_file, index=False)
cleaned_size = os.path.getsize(output_file) / (1024**2)
print(f"Cleaned file size: {cleaned_size:.2f} MB")

# Final summary
print("\n" + "="*70)
print("CLEANING SUMMARY")
print("="*70)
print(f"Original row count:     {original_size:,}MB")
print(f"Final row count:        {len(df):,} rows")
print(f"Total rows removed:     {2589083 - len(df):,} rows ({((2589083-len(df))/2589083)*100:.2f}%)")
print(f"Columns in final dataset: {len(df.columns)}")
print(f"\nFinal dataset shape: {df.shape}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nData saved successfully to: {output_file}")
print("="*70)
