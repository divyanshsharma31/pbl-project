"""
Simplified Direct Training Script
Creates models directly without class overhead
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

print("TRAINING SPATIAL-TEMPORAL AQI MODELS")
print("="*70)

try:
    print("\nStep 1: Loading data...")
    # Read directly with limited columns
    df = pd.read_csv('files/station_hour_cleaned.csv', 
                     usecols=['StationName', 'City', 'State', 'Datetime', 'Year', 'Month',
                             'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3',
                             'Season', 'Day_period', 'AQI'])
    print("[OK] Loaded {} records".format(len(df)))
    
    # Sample for training (use 20% random sample)
    df = df.sample(frac=0.2, random_state=42)
    print("[OK] Using {} records for training".format(len(df)))
    
    print("\nStep 2: Encoding categorical variables...")
    le_station = LabelEncoder()
    le_city = LabelEncoder()
    le_state = LabelEncoder()
    le_season = LabelEncoder()
    le_period = LabelEncoder()
    
    df['StationName_enc'] = le_station.fit_transform(df['StationName'].astype(str))
    df['City_enc'] = le_city.fit_transform(df['City'].astype(str))
    df['State_enc'] = le_state.fit_transform(df['State'].astype(str))
    df['Season_enc'] = le_season.fit_transform(df['Season'].astype(str))
    df['Period_enc'] = le_period.fit_transform(df['Day_period'].astype(str))
    
    print("[OK] Encoded categorical variables")
    
    print("\nStep 3: Preparing features...")
    feature_cols = ['Year', 'Month', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3',
                   'StationName_enc', 'City_enc', 'State_enc', 'Season_enc', 'Period_enc']
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['AQI']
    
    print("[OK] Features shape: {}".format(X.shape))
    
    print("\nStep 4: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("[OK] Train: {}, Test: {}".format(len(X_train), len(X_test)))
    
    print("\nStep 5: Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[OK] Features scaled")
    
    print("\nStep 6: Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1, verbose=1)
    rf.fit(X_train_scaled, y_train)
    rf_score = r2_score(y_test, rf.predict(X_test_scaled))
    print("[OK] RF R2 Score: {:.4f}".format(rf_score))
    
    print("\nStep 7: Training Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=30, max_depth=4, random_state=42, verbose=1)
    gb.fit(X_train_scaled, y_train)
    gb_score = r2_score(y_test, gb.predict(X_test_scaled))
    print("[OK] GB R2 Score: {:.4f}".format(gb_score))
    
    print("\nStep 8: Computing location statistics...")
    location_stats = {}
    for station in df['StationName'].unique():
        s_data = df[df['StationName'] == station]
        location_stats[station] = {
            'city': s_data['City'].iloc[0],
            'avg_aqi': float(s_data['AQI'].mean()),
            'std_aqi': float(s_data['AQI'].std()),
            'records': len(s_data)
        }
    print("[OK] Computed stats for {} stations".format(len(location_stats)))
    
    print("\nStep 9: Creating city-location mapping...")
    city_mapping = {}
    for city in df['City'].unique():
        city_mapping[city] = df[df['City'] == city]['StationName'].unique().tolist()
    print("[OK] {} cities mapped".format(len(city_mapping)))
    
    print("\nStep 10: Saving models...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(rf, 'models/random_forest_model.pkl')
    joblib.dump(gb, 'models/gradient_boosting_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(feature_cols, 'models/feature_names.pkl')
    joblib.dump(location_stats, 'models/location_stats.pkl')
    joblib.dump(city_mapping, 'models/city_location_mapping.pkl')
    joblib.dump({
        'station': le_station,
        'city': le_city,
        'state': le_state,
        'season': le_season,
        'period': le_period
    }, 'models/label_encoders.pkl')
    
    print("[OK] All models saved!")
    
    print("\n" + "="*70)
    print("SUCCESS! Models trained and saved.")
    print("="*70)
    print("\nRun the Streamlit app:")
    print("streamlit run app_enhanced.py\n")
    
except Exception as e:
    print("\nERROR: {}".format(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
