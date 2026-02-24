"""
Train and Save Spatial-Temporal AQI Models
Run this script once to train models and save them
"""

import sys
from aqi_spatial_temporal_model import SpatialTemporalAQIPredictor
from aqi_prevention_guide import AQIPreventionGuide

def train_models():
    """Train and save spatial-temporal models"""
    
    print("="*70)
    print("SPATIAL-TEMPORAL AQI PREDICTION MODEL TRAINING")
    print("="*70)
    
    # Initialize predictor
    predictor = SpatialTemporalAQIPredictor(data_path='files/station_hour_cleaned.csv')
    
    # Load and prepare data
    print("\n1️⃣ Loading and preparing data...")
    df = predictor.load_and_prepare_data()
    print(f"   ✓ Data loaded: {len(df)} records")
    print(f"   ✓ Cities: {df['City'].nunique()}, Stations: {df['StationName'].nunique()}")
    
    # Train models
    print("\n2️⃣ Training spatial-temporal models...")
    metrics = predictor.train_models(test_size=0.2)
    
    print("\n   Model Performance:")
    for model_name, scores in metrics.items():
        print(f"   {model_name}:")
        print(f"     - R² Score: {scores['r2']:.4f}")
        print(f"     - RMSE: {scores['rmse']:.2f}")
    
    # Save models
    print("\n3️⃣ Saving trained models...")
    try:
        predictor.save_models(path='models/')
        print("   ✓ Models saved successfully")
    except Exception as e:
        print(f"   ✗ Error saving models: {e}")
        return False
    
    # Display statistics
    print("\n4️⃣ Location Statistics:")
    cities = predictor.get_all_cities()
    print(f"   ✓ Total Cities: {len(cities)}")
    
    for city in cities[:5]:
        stations = predictor.get_city_stations(city)
        print(f"     - {city}: {len(stations)} monitoring stations")
    
    # Sample predictions
    print("\n5️⃣ Sample Predictions:")
    if cities:
        sample_city = cities[0]
        sample_stations = predictor.get_city_stations(sample_city)
        if sample_stations:
            sample_station = sample_stations[0]
            pred = predictor.predict_spatial_temporal_aqi(sample_station, 2024, 12)
            if pred:
                print(f"   Station: {pred['station']}")
                print(f"   City: {pred['city']}")
                print(f"   Predicted AQI (Dec 2024): {pred['predicted_aqi']:.1f}")
                print(f"   Confidence: {pred['confidence']}")
    
    # Test prevention guide
    print("\n6️⃣ Testing Prevention Guide...")
    guide = AQIPreventionGuide()
    
    test_aqi_values = [25, 75, 125, 175, 250, 350]
    for aqi in test_aqi_values:
        category = guide.get_aqi_category(aqi)
        print(f"   AQI {aqi:3d} → {category}")
    
    # High-risk and safe locations
    print("\n7️⃣ Identifying High-Risk and Safe Locations...")
    
    if cities:
        sample_city = cities[0]
        high_risk = predictor.get_high_risk_locations(sample_city, threshold=150)
        safe = predictor.get_safe_locations(sample_city, threshold=100)
        
        print(f"   {sample_city}:")
        print(f"     - High-Risk Locations: {len(high_risk)}")
        print(f"     - Safe Locations: {len(safe)}")
    
    print("\n" + "="*70)
    print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nModels are ready for use in the Streamlit app")
    print("Run: streamlit run app_enhanced.py")
    
    return True


if __name__ == "__main__":
    try:
        success = train_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
