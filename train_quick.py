"""
Quick Training Script for Spatial-Temporal AQI Models
Uses stratified sampling for faster training
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SPATIAL-TEMPORAL AQI MODEL TRAINING (QUICK VERSION)")
print("="*70)

try:
    print("\n1. Loading cleaned dataset...")
    # Load data in chunks and sample for faster training
    df_chunks = pd.read_csv('files/station_hour_cleaned.csv', chunksize=100000)
    dfs = []
    for i, chunk in enumerate(df_chunks):
        # Sample 10% from each chunk for faster training
        sampled = chunk.sample(frac=0.1, random_state=42)
        dfs.append(sampled)
        print(f"   Loaded chunk {i+1}")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"   ✓ Total records: {len(df)}")
    print(f"   ✓ Cities: {df['City'].nunique()}, Stations: {df['StationName'].nunique()}")
    
    print("\n2. Preparing features...")
    # Convert Datetime
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # Extract temporal features
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['Hour'] = df['Datetime'].dt.hour
    df['Quarter'] = df['Datetime'].dt.quarter
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['DayOfYear'] = df['Datetime'].dt.dayofyear
    
    #  Prepare features
    label_encoders = {}
    categorical_cols = ['StationName', 'City', 'State', 'Season', 'Day_period']
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    feature_cols = [
        'Year', 'Month', 'Day', 'Hour', 'Quarter', 'DayOfWeek', 'DayOfYear',
        'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3',
        'StationName', 'City', 'State', 'Season', 'Day_period'
    ]
    
    X = df_encoded[feature_cols].fillna(df_encoded[feature_cols].mean())
    y = df_encoded['AQI']
    
    print(f"   ✓ Features prepared: {X.shape}")
    
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   ✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✓ Scaling complete")
    
    print("\n5. Training models...")
    models = {}
    
    # Random Forest
    print("   Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=50,  # Reduced from 100
        max_depth=10,  # Reduced from 15
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    models['random_forest'] = rf_model
    print(f"     R² = {rf_r2:.4f}, RMSE = {rf_rmse:.2f}")
    
    # Gradient Boosting
    print("   Training Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=50,  # Reduced from 100
        learning_rate=0.05,
        max_depth=4,  # Reduced from 5
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    models['gradient_boosting'] = gb_model
    print(f"     R² = {gb_r2:.4f}, RMSE = {gb_rmse:.2f}")
    
    print("\n6. Saving models...")
    os.makedirs('models', exist_ok=True)
    
    for model_name, model in models.items():
        joblib.dump(model, f'models/{model_name}_model.pkl')
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(feature_cols, 'models/feature_names.pkl')
    
    # Save location stats
    location_stats = {}
    for station in df['StationName'].unique():
        station_data = df[df['StationName'] == station]
        location_stats[station] = {
            'city': station_data['City'].iloc[0],
            'avg_aqi': station_data['AQI'].mean(),
            'std_aqi': station_data['AQI'].std(),
            'min_aqi': station_data['AQI'].min(),
            'max_aqi': station_data['AQI'].max(),
            'avg_pm25': station_data['PM2.5'].mean(),
            'avg_pm10': station_data['PM10'].mean(),
            'records': len(station_data)
        }
    
    joblib.dump(location_stats, 'models/location_stats.pkl')
    
    # Save city-location mapping
    city_location_mapping = {}
    for city in df['City'].unique():
        stations = df[df['City'] == city]['StationName'].unique().tolist()
        city_location_mapping[city] = stations
    
    joblib.dump(city_location_mapping, 'models/city_location_mapping.pkl')
    
    print("   ✓ All models saved to 'models/' directory")
    
    print("\n7. Model Summary:")
    print(f"   Random Forest: R² = {rf_r2:.4f}")
    print(f"   Gradient Boosting: R² = {gb_r2:.4f}")
    print(f"   Cities: {len(city_location_mapping)}")
    print(f"   Stations: {len(location_stats)}")
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nYou can now run the Streamlit app:")
    print(">>> streamlit run app_enhanced.py")
    print("\nNote: These models are trained on 10% sampled data for speed.")
    print("For better accuracy, run: python train_full_models.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
