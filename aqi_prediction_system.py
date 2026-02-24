"""
Random Forest Model for AQI Prediction & Improvement Measures
Predicts AQI category (Safe/Hazardous) and suggests measures to improve air quality
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===== AQI CATEGORY & HEALTH EFFECTS DATABASE =====
AQI_EFFECTS_DB = {
    'Good': {
        'range': '0-50',
        'color': '🟢 GREEN',
        'status': 'SAFE',
        'health_effects': [
            '✓ No adverse health effects',
            '✓ Excellent air quality',
            '✓ Safe for all outdoor activities',
            '✓ No restrictions for any population group'
        ],
        'precautions': 'Enjoy outdoor activities',
        'severity': 1
    },
    'Satisfactory': {
        'range': '51-100',
        'color': '🟡 YELLOW',
        'status': 'SAFE',
        'health_effects': [
            '⚠ Slight respiratory discomfort in sensitive groups',
            '⚠ Minimal risk for general population',
            '⚠ Asthma symptoms may appear occasionally'
        ],
        'precautions': 'Sensitive individuals should limit prolonged outdoor activities',
        'severity': 2
    },
    'Moderately Polluted': {
        'range': '101-150',
        'color': '🟠 ORANGE',
        'status': 'MODERATE HAZARD',
        'health_effects': [
            '⛔ Respiratory discomfort for sensitive groups',
            '⛔ Increased asthma attacks and coughing',
            '⛔ Difficulty in breathing for children & elderly',
            '⛔ General fatigue and reduced physical performance'
        ],
        'precautions': 'Children, elderly, & those with respiratory disease should avoid outdoor activities',
        'severity': 3
    },
    'Poor': {
        'range': '151-200',
        'color': '🔴 RED',
        'status': 'HAZARDOUS',
        'health_effects': [
            '🚨 Severe respiratory illness in general population',
            '🚨 Increased heart disease risk',
            '🚨 Exacerbated asthma and cardiovascular diseases',
            '🚨 Reduced lung function and physical capacity',
            '🚨 Hospital admissions likely'
        ],
        'precautions': 'Most people should stay indoors; use N95/N100 masks if going outside',
        'severity': 4
    },
    'Very Poor': {
        'range': '201-300',
        'color': '🟣 PURPLE',
        'status': 'HAZARDOUS',
        'health_effects': [
            '🚨 Life-threatening respiratory illness',
            '🚨 Severe cardiovascular complications',
            '🚨 Mass respiratory symptoms expected',
            '🚨 Hospital emergencies and mortality risk',
            '🚨 Work capacity severely reduced'
        ],
        'precautions': 'All groups should stay indoors with air filters ON; emergency preparedness essential',
        'severity': 5
    },
    'Severe': {
        'range': '301+',
        'color': '🟫 MAROON',
        'status': 'HAZARDOUS',
        'health_effects': [
            '💀 Life-threatening conditions for all',
            '💀 Respiratory failure and cardiac arrest risk',
            '💀 Mass casualties possible',
            '💀 Immediate medical intervention required',
            '💀 Irreversible lung damage possible'
        ],
        'precautions': 'Total lockdown recommended; stay indoors with advanced air filtration; call emergency services',
        'severity': 6
    }
}

IMPROVEMENT_MEASURES = {
    'PM2.5': {
        'priority': 'CRITICAL',
        'measures': [
            '1. Install air purifiers and HEPA filters in homes',
            '2. Reduce industrial emissions enforcement',
            '3. Promote electric vehicles over diesel/petrol',
            '4. Control construction and dust generation',
            '5. Implement stricter pollution standards',
            '6. Ban stubble burning in agricultural areas'
        ]
    },
    'PM10': {
        'priority': 'CRITICAL',
        'measures': [
            '1. Street sweeping and wet cleaning of roads',
            '2. Dust control at construction sites',
            '3. Control unpaved road traffic',
            '4. Water spraying to reduce dust particles',
            '5. Vegetative cover on open ground'
        ]
    },
    'Vehicle Count': {
        'priority': 'HIGH',
        'measures': [
            '1. Promote public transportation (buses, metro)',
            '2. Encourage carpooling and ride-sharing',
            '3. Implement odd-even vehicle schemes',
            '4. Work from home policies',
            '5. Improve road infrastructure for better flow',
            '6. Implement electric vehicle incentives'
        ]
    },
    'Industrial Activity Index': {
        'priority': 'HIGH',
        'measures': [
            '1. Enforce green manufacturing practices',
            '2. Install scrubbers on industrial chimneys',
            '3. Shift polluting industries outside urban areas',
            '4. Implement industry-wise pollution limits',
            '5. Regular stack emission monitoring',
            '6. Promote cleaner production technologies'
        ]
    },
    'NO2': {
        'priority': 'HIGH',
        'measures': [
            '1. Control vehicle emissions at source',
            '2. Improve fuel quality standards',
            '3. Regular vehicle maintenance campaigns',
            '4. Reduce diesel vehicle usage',
            '5. Industrial NOx control measures'
        ]
    },
    'CO': {
        'priority': 'MEDIUM',
        'measures': [
            '1. Improve vehicle fuel efficiency',
            '2. Regular emission testing',
            '3. Promote renewable energy usage',
            '4. Better traffic management',
            '5. Control residential fuel burning'
        ]
    },
    'SO2': {
        'priority': 'MEDIUM',
        'measures': [
            '1. Use low-sulfur fuel in industries',
            '2. Industrial desulfurization equipment',
            '3. Power plant emission control',
            '4. Monitor and regulate combustion sources',
            '5. Implement sulfur emission standards'
        ]
    },
    'O3': {
        'priority': 'MEDIUM',
        'measures': [
            '1. Reduce NOx and VOC emissions',
            '2. Control industrial emissions',
            '3. Green cover enhancement',
            '4. Regulate secondary pollutant formation'
        ]
    },
    'Wind Speed (km/h)': {
        'priority': 'SUPPORTIVE',
        'measures': [
            '1. Plant trees as windbreaks',
            '2. Urban forest development',
            '3. Monitor wind patterns for pollution dispersal',
            '4. Plan outdoor activities during high wind days'
        ]
    },
    'Temperature (°C)': {
        'priority': 'SUPPORTIVE',
        'measures': [
            '1. Increase urban green cover',
            '2. Cool pavement technology',
            '3. Green roofs and walls',
            '4. Reduce heat island effect',
            '5. Tree plantation programs'
        ]
    }
}

class AQIPredictionSystem:
    def __init__(self):
        self.rf_classifier = None
        self.rf_regressor = None
        self.feature_names = None
        self.scaler = None
        
    def categorize_aqi(self, aqi_value):
        """Categorize AQI value into category"""
        if aqi_value <= 50:
            return 'Good'
        elif aqi_value <= 100:
            return 'Satisfactory'
        elif aqi_value <= 150:
            return 'Moderately Polluted'
        elif aqi_value <= 200:
            return 'Poor'
        elif aqi_value <= 300:
            return 'Very Poor'
        else:
            return 'Severe'
    
    def train(self):
        """Train the Random Forest models"""
        print("\n" + "="*100)
        print("TRAINING RANDOM FOREST MODEL FOR AQI PREDICTION")
        print("="*100)
        
        # Load data
        df = pd.read_csv('files/indian_aqi_realistic_2019_2024.csv')
        
        # Add category column
        df['AQI_Category'] = df['AQI'].apply(self.categorize_aqi)
        
        # Features
        self.feature_names = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 
                             'Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 
                             'Rainfall (mm)', 'Pressure (hPa)', 'Vehicle Count', 
                             'Industrial Activity Index']
        
        X = df[self.feature_names]
        y_category = df['AQI_Category']
        y_aqi = df['AQI']
        
        # Train classifier
        X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2, random_state=42)
        
        self.rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        self.rf_classifier.fit(X_train, y_train)
        accuracy = self.rf_classifier.score(X_test, y_test)
        print(f"\n✓ Classifier Accuracy: {accuracy:.4f}")
        
        # Train regressor
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_aqi, test_size=0.2, random_state=42)
        
        self.rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        self.rf_regressor.fit(X_train_reg, y_train_reg)
        r2 = self.rf_regressor.score(X_test_reg, y_test_reg)
        print(f"✓ Regressor R² Score: {r2:.4f}")
        
        self.print_feature_importance()
    
    def print_feature_importance(self):
        """Print feature importance and improvement measures"""
        print("\n" + "="*100)
        print("FEATURE IMPORTANCE - RECOMMENDED MEASURES TO IMPROVE AQI")
        print("="*100)
        
        importances = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.rf_regressor.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop Contributing Factors to AQI (in order of impact):\n")
        for idx, (_, row) in enumerate(importances.iterrows(), 1):
            feature = row['Feature']
            importance = row['Importance']
            measures = IMPROVEMENT_MEASURES.get(feature, {'priority': 'MONITOR', 'measures': ['Keep monitoring']})
            
            print(f"\n{idx}. {feature.upper()} (Importance: {importance:.4f}) - Priority: {measures['priority']}")
            for measure in measures['measures']:
                print(f"   {measure}")
    
    def predict_aqi_category(self, pm25, pm10, no2, co, so2, o3, temp, humidity, wind, rainfall, pressure, vehicles, industrial):
        """Predict AQI category and show health effects"""
        input_data = np.array([[pm25, pm10, no2, co, so2, o3, temp, humidity, wind, rainfall, pressure, vehicles, industrial]])
        
        # Predict category
        predicted_category = self.rf_classifier.predict(input_data)[0]
        
        # Predict AQI value
        predicted_aqi = self.rf_regressor.predict(input_data)[0]
        
        return predicted_aqi, predicted_category
    
    def display_prediction(self, city, aqi_value, category):
        """Display detailed prediction with health effects"""
        if category not in AQI_EFFECTS_DB:
            category = 'Satisfactory'
        
        effects = AQI_EFFECTS_DB[category]
        
        print("\n" + "="*100)
        print(f"AQI PREDICTION FOR {city.upper()}")
        print("="*100)
        print(f"\n📊 Predicted AQI Value: {aqi_value:.1f} {effects['color']}")
        print(f"📌 Category: {category} (Range: {effects['range']})")
        print(f"🔴 Status: {effects['status']}")
        
        print(f"\n⚠ HEALTH EFFECTS:")
        for effect in effects['health_effects']:
            print(f"   {effect}")
        
        print(f"\n🛡 PRECAUTIONS:")
        print(f"   → {effects['precautions']}")
        
        print("\n" + "="*100)
    
    def get_improvement_recommendations(self, top_n=5):
        """Get top N recommendations to improve AQI"""
        importances = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.rf_regressor.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n" + "="*100)
        print("TOP RECOMMENDATIONS TO IMPROVE AQI")
        print("="*100)
        
        for idx, (_, row) in enumerate(importances.head(top_n).iterrows(), 1):
            feature = row['Feature']
            importance = row['Importance']
            measures = IMPROVEMENT_MEASURES.get(feature, {'priority': 'MONITOR', 'measures': []})
            
            print(f"\n🎯 PRIORITY {idx}: {feature.upper()} ({measures['priority']})")
            print(f"   Impact Score: {importance:.4f}")
            for measure in measures['measures'][:3]:  # Top 3 measures
                print(f"   • {measure}")


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
    system = AQIPredictionSystem()
    system.train()
    
    # Example prediction
    print("\n" + "="*100)
    print("EXAMPLE PREDICTION")
    print("="*100)
    
    # Sample data for a city with high pollution
    aqi, category = system.predict_aqi_category(
        pm25=85, pm10=150, no2=65, co=1.2, so2=25, o3=45,
        temp=28, humidity=60, wind=5, rainfall=0, pressure=1010,
        vehicles=150000, industrial=7.5
    )
    
    system.display_prediction("New Delhi", aqi, category)
    
    # Get recommendations
    system.get_improvement_recommendations(top_n=5)
    
    print("\n✓ Model is ready for predictions!")
