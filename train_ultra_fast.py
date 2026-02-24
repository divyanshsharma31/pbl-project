"""
ULTRA-FAST TRAINING - AQI Models
Uses only 5% data for immediate testing
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

print("ULTRA-FAST MODEL TRAINING")
print("="*50)

try:
    # Load minimal data
    print("Loading data (5% sample)...")
    usecols = ['StationName', 'City', 'Month', 'Year', 'PM2.5', 'PM10', 'AQI']
    df = pd.read_csv('files/station_hour_cleaned.csv', usecols=usecols)
    df = df.sample(frac=0.05, random_state=42)
    print(f"[OK] {len(df)} records")
    
    # Quick encoding
    print("Preparing features...")
    le_station = LabelEncoder()
    le_city = LabelEncoder()
    
    df['Station_enc'] = le_station.fit_transform(df['StationName'].astype(str))
    df['City_enc'] = le_city.fit_transform(df['City'].astype(str))
    
    X = df[['Year', 'Month', 'PM2.5', 'PM10', 'Station_enc', 'City_enc']].fillna(df[['Year', 'Month', 'PM2.5', 'PM10', 'Station_enc', 'City_enc']].mean())
    y = df['AQI']
    
    print(f"[OK] Features ready")
    
    # Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[OK] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale
    print("Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[OK] Scaled")
    
    # Train RF
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=10, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_score = rf.score(X_test_scaled, y_test)
    print(f"[OK] RF Score: {rf_score:.4f}")
    
    # Train GB
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=42)
    gb.fit(X_train_scaled, y_train)
    gb_score = gb.score(X_test_scaled, y_test)
    print(f"[OK] GB Score: {gb_score:.4f}")
    
    # Save models
    print("Saving models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/random_forest_model.pkl')
    joblib.dump(gb, 'models/gradient_boosting_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(['Year', 'Month', 'PM2.5', 'PM10', 'Station_enc', 'City_enc'], 'models/feature_names.pkl')
    
    # Location stats
    location_stats = {}
    for station in df['StationName'].unique():
        s_data = df[df['StationName'] == station]
        location_stats[station] = {
            'city': s_data['City'].iloc[0],
            'avg_aqi': float(s_data['AQI'].mean()),
            'std_aqi': float(s_data['AQI'].std()),
            'records': len(s_data)
        }
    joblib.dump(location_stats, 'models/location_stats.pkl')
    
    # City mapping
    city_mapping = {}
    for city in df['City'].unique():
        city_mapping[city] = df[df['City'] == city]['StationName'].unique().tolist()
    joblib.dump(city_mapping, 'models/city_location_mapping.pkl')
    
    joblib.dump({'station': le_station, 'city': le_city}, 'models/label_encoders.pkl')
    
    print("[OK] Models saved!")
    print("\n" + "="*50)
    print("SUCCESS! Ready to run app")
    print("="*50)
    print("\nRun: streamlit run app_enhanced.py\n")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
