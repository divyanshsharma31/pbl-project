"""
Spatial-Temporal AQI Prediction Model
Predicts AQI by location (station) and time period (year/month)
Uses ensemble methods and incorporates spatial clustering for better predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class SpatialTemporalAQIPredictor:
    """
    Predicts AQI based on spatial (location/station) and temporal (year/month) features
    """
    
    def __init__(self, data_path='files/station_hour_cleaned.csv'):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.location_stats = {}
        self.city_location_mapping = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for spatial-temporal modeling"""
        print("Loading cleaned dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert Datetime to datetime
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        
        # Extract temporal features
        self.df['Year'] = self.df['Datetime'].dt.year
        self.df['Month'] = self.df['Datetime'].dt.month
        self.df['Day'] = self.df['Datetime'].dt.day
        self.df['Hour'] = self.df['Datetime'].dt.hour
        self.df['Quarter'] = self.df['Datetime'].dt.quarter
        self.df['DayOfWeek'] = self.df['Datetime'].dt.dayofweek
        self.df['DayOfYear'] = self.df['Datetime'].dt.dayofyear
        
        # Create location mapping (City -> List of Stations)
        self._create_location_mapping()
        
        # Calculate location-specific statistics
        self._calculate_location_stats()
        
        print(f"Data loaded: {len(self.df)} records")
        print(f"Cities: {self.df['City'].nunique()}, Stations: {self.df['StationName'].nunique()}")
        
        return self.df
    
    def _create_location_mapping(self):
        """Create mapping of cities to monitoring stations"""
        for city in self.df['City'].unique():
            stations = self.df[self.df['City'] == city]['StationName'].unique().tolist()
            self.city_location_mapping[city] = stations
    
    def _calculate_location_stats(self):
        """Calculate statistics for each location"""
        location_groups = self.df.groupby('StationName').agg({
            'AQI': ['mean', 'std', 'min', 'max', 'count'],
            'PM2.5': 'mean',
            'PM10': 'mean',
            'City': 'first'
        }).round(2)
        
        for station in self.df['StationName'].unique():
            station_data = self.df[self.df['StationName'] == station]
            self.location_stats[station] = {
                'city': station_data['City'].iloc[0],
                'avg_aqi': station_data['AQI'].mean(),
                'std_aqi': station_data['AQI'].std(),
                'min_aqi': station_data['AQI'].min(),
                'max_aqi': station_data['AQI'].max(),
                'avg_pm25': station_data['PM2.5'].mean(),
                'avg_pm10': station_data['PM10'].mean(),
                'records': len(station_data)
            }
    
    def train_models(self, test_size=0.2):
        """Train spatial-temporal models"""
        print("\nTraining spatial-temporal models...")
        
        # Prepare features
        X, y = self._prepare_features()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("Training Random Forest Regressor...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        print(f"  RF R² Score: {rf_r2:.4f}, RMSE: {rf_rmse:.2f}")
        self.models['random_forest'] = rf_model
        
        # Train Gradient Boosting
        print("Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_r2 = r2_score(y_test, gb_pred)
        gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
        print(f"  GB R² Score: {gb_r2:.4f}, RMSE: {gb_rmse:.2f}")
        self.models['gradient_boosting'] = gb_model
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return {
            'random_forest': {'r2': rf_r2, 'rmse': rf_rmse},
            'gradient_boosting': {'r2': gb_r2, 'rmse': gb_rmse}
        }
    
    def _prepare_features(self):
        """Prepare features for modeling"""
        df_copy = self.df.copy()
        
        # Encode categorical variables
        categorical_cols = ['StationName', 'City', 'State', 'Season', 'Day_period']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col].astype(str))
            else:
                df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))
        
        # Select features
        feature_cols = [
            'Year', 'Month', 'Day', 'Hour', 'Quarter', 'DayOfWeek', 'DayOfYear',
            'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3',
            'StationName', 'City', 'State', 'Season', 'Day_period'
        ]
        
        X = df_copy[feature_cols].fillna(df_copy[feature_cols].mean())
        y = df_copy['AQI']
        
        return X, y
    
    def predict_spatial_temporal_aqi(self, station_name, year, month, use_ensemble=True):
        """
        Predict AQI for a specific station at a given year and month
        
        Args:
            station_name: Name of the monitoring station
            year: Prediction year
            month: Prediction month (1-12)
            use_ensemble: If True, average predictions from all models
        
        Returns:
            dict with prediction and confidence info
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")
        
        try:
            # Create feature vector
            sample_data = self.df[self.df['StationName'] == station_name].iloc[0].copy()
            
            # Update temporal features
            sample_data['Year'] = year
            sample_data['Month'] = month
            sample_data['Day'] = 15  # Use middle of month
            sample_data['Hour'] = 12  # Use midday
            sample_data['Quarter'] = (month - 1) // 3 + 1
            
            # Prepare feature vector
            feature_values = []
            for feature in self.feature_names:
                if feature in sample_data.index:
                    value = sample_data[feature]
                else:
                    # Use average for missing features
                    value = self.df[feature].mean()
                feature_values.append(value)
            
            X_new = np.array(feature_values).reshape(1, -1)
            X_new_scaled = self.scaler.transform(X_new)
            
            # Get predictions
            predictions = {}
            for model_name, model in self.models.items():
                pred = model.predict(X_new_scaled)[0]
                predictions[model_name] = max(0, pred)  # Ensure non-negative
            
            if use_ensemble:
                ensemble_pred = np.mean(list(predictions.values()))
            else:
                ensemble_pred = predictions['random_forest']
            
            # Get location statistics for context
            location_info = self.location_stats.get(station_name, {})
            
            return {
                'station': station_name,
                'city': location_info.get('city', 'Unknown'),
                'year': year,
                'month': month,
                'predicted_aqi': round(ensemble_pred, 2),
                'rf_prediction': round(predictions['random_forest'], 2),
                'gb_prediction': round(predictions['gradient_boosting'], 2),
                'historical_avg': round(location_info.get('avg_aqi', 0), 2),
                'historical_std': round(location_info.get('std_aqi', 0), 2),
                'confidence': 'High' if abs(ensemble_pred - location_info.get('avg_aqi', 0)) < 100 else 'Medium'
            }
        
        except Exception as e:
            print(f"Error predicting for {station_name}: {str(e)}")
            return None
    
    def predict_city_aqi(self, city_name, year, month):
        """
        Predict AQI for all stations in a city for a given year and month
        
        Returns:
            DataFrame with predictions for all stations in the city
        """
        if city_name not in self.city_location_mapping:
            return pd.DataFrame()
        
        stations = self.city_location_mapping[city_name]
        predictions = []
        
        for station in stations:
            pred = self.predict_spatial_temporal_aqi(station, year, month)
            if pred:
                predictions.append(pred)
        
        return pd.DataFrame(predictions)
    
    def get_high_risk_locations(self, city=None, threshold=200):
        """
        Identify locations with AQI higher than threshold
        
        Args:
            city: Filter by city (None for all)
            threshold: AQI threshold (default 200 = Poor)
        
        Returns:
            DataFrame of high-risk locations
        """
        high_risk = []
        
        for station, stats in self.location_stats.items():
            if city and stats['city'] != city:
                continue
            
            if stats['avg_aqi'] > threshold:
                high_risk.append({
                    'station': station,
                    'city': stats['city'],
                    'avg_aqi': stats['avg_aqi'],
                    'risk_level': 'EXTREME' if stats['avg_aqi'] > 300 else 'HIGH',
                    'recommendation': 'AVOID' if stats['avg_aqi'] > 300 else 'MINIMIZE VISITS'
                })
        
        if high_risk:
            return pd.DataFrame(high_risk).sort_values('avg_aqi', ascending=False)
        return pd.DataFrame()
    
    def get_safe_locations(self, city=None, threshold=100):
        """
        Identify locations with AQI lower than threshold
        
        Args:
            city: Filter by city (None for all)
            threshold: AQI threshold for "safe" (default 100 = Satisfactory)
        
        Returns:
            DataFrame of safe locations
        """
        safe = []
        
        for station, stats in self.location_stats.items():
            if city and stats['city'] != city:
                continue
            
            if stats['avg_aqi'] < threshold:
                safe.append({
                    'station': station,
                    'city': stats['city'],
                    'avg_aqi': stats['avg_aqi'],
                    'air_quality': 'EXCELLENT' if stats['avg_aqi'] < 50 else 'GOOD',
                    'recommendation': 'RECOMMENDED'
                })
        
        if safe:
            return pd.DataFrame(safe).sort_values('avg_aqi', ascending=True)
        return pd.DataFrame()
    
    def get_all_cities(self):
        """Get list of all cities"""
        return sorted(list(self.city_location_mapping.keys()))
    
    def get_city_stations(self, city):
        """Get all stations in a city"""
        return self.city_location_mapping.get(city, [])
    
    def save_models(self, path='models/'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for model_name, model in self.models.items():
            joblib.dump(model, f'{path}{model_name}_model.pkl')
        
        joblib.dump(self.scaler, f'{path}scaler.pkl')
        joblib.dump(self.label_encoders, f'{path}label_encoders.pkl')
        joblib.dump(self.location_stats, f'{path}location_stats.pkl')
        joblib.dump(self.city_location_mapping, f'{path}city_location_mapping.pkl')
        joblib.dump(self.feature_names, f'{path}feature_names.pkl')
        
        print(f"Models saved to {path}")
    
    def load_models(self, path='models/'):
        """Load trained models"""
        for model_name in ['random_forest', 'gradient_boosting']:
            self.models[model_name] = joblib.load(f'{path}{model_name}_model.pkl')
        
        self.scaler = joblib.load(f'{path}scaler.pkl')
        self.label_encoders = joblib.load(f'{path}label_encoders.pkl')
        self.location_stats = joblib.load(f'{path}location_stats.pkl')
        self.city_location_mapping = joblib.load(f'{path}city_location_mapping.pkl')
        self.feature_names = joblib.load(f'{path}feature_names.pkl')
        
        print(f"Models loaded from {path}")


if __name__ == "__main__":
    # Example usage
    predictor = SpatialTemporalAQIPredictor()
    predictor.load_and_prepare_data()
    metrics = predictor.train_models()
    
    # Predict for a station
    pred = predictor.predict_spatial_temporal_aqi('StationName', 2024, 12)
    print(pred)
