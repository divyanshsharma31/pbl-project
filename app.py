"""
Streamlit App for Random Forest AQI Prediction System
Interactive web application for AQI prediction with health effects and improvement measures
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .safe {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .warning {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .danger {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .info-text {
        font-size: 14px;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEALTH EFFECTS DATABASE =====
HEALTH_EFFECTS_DB = {
    'Good': {
        'range': '0-50',
        'emoji': '🟢',
        'status': 'SAFE',
        'severity': 1,
        'color': '#28a745',
        'health_effects': [
            '✓ No adverse health effects',
            '✓ Excellent air quality for all activities',
            '✓ Safe for all population groups'
        ],
        'precautions': 'Enjoy outdoor activities freely'
    },
    'Satisfactory': {
        'range': '51-100',
        'emoji': '🟡',
        'status': 'SAFE',
        'severity': 2,
        'color': '#ffc107',
        'health_effects': [
            '⚠ Minor respiratory symptoms in sensitive groups',
            '⚠ Minimal risk for general population',
            '⚠ Occasional asthma symptoms'
        ],
        'precautions': 'Sensitive individuals should limit prolonged outdoor exposure'
    },
    'Moderately Polluted': {
        'range': '101-150',
        'emoji': '🟠',
        'status': 'MODERATE HAZARD',
        'severity': 3,
        'color': '#fd7e14',
        'health_effects': [
            '⛔ Respiratory discomfort for sensitive groups',
            '⛔ Increased asthma attacks and coughing',
            '⛔ Difficulty breathing for children & elderly'
        ],
        'precautions': 'Children, elderly, & people with respiratory disease should avoid outdoor activities'
    },
    'Poor': {
        'range': '151-200',
        'emoji': '🔴',
        'status': 'HAZARDOUS',
        'severity': 4,
        'color': '#dc3545',
        'health_effects': [
            '🚨 Severe respiratory illness in general population',
            '🚨 Increased heart disease risk',
            '🚨 Hospital admissions likely to increase'
        ],
        'precautions': 'Most people stay indoors; Use N95 masks if outside'
    },
    'Very Poor': {
        'range': '201-300',
        'emoji': '🟣',
        'status': 'HAZARDOUS',
        'severity': 5,
        'color': '#6f42c1',
        'health_effects': [
            '🚨 Life-threatening respiratory illness',
            '🚨 Severe cardiovascular complications',
            '🚨 Hospital emergencies & mortality risk'
        ],
        'precautions': 'All should stay indoors with air filters; Emergency preparedness essential'
    },
    'Severe': {
        'range': '301+',
        'emoji': '💀',
        'status': 'HAZARDOUS',
        'severity': 6,
        'color': '#721c24',
        'health_effects': [
            '💀 Life-threatening conditions for ALL',
            '💀 Respiratory failure and cardiac arrest risk',
            '💀 Mass casualties and mortality possible'
        ],
        'precautions': 'Total lockdown; Stay indoors with advanced air filtration'
    }
}

IMPROVEMENT_MEASURES = {
    'PM2.5': {
        'priority': 'CRITICAL',
        'measures': [
            '✓ Install air purifiers and HEPA filters in homes',
            '✓ Reduce industrial emissions enforcement',
            '✓ Promote electric vehicles over diesel/petrol',
            '✓ Control construction dust generation',
            '✓ Implement stricter pollution standards'
        ]
    },
    'PM10': {
        'priority': 'CRITICAL',
        'measures': [
            '✓ Street sweeping and wet cleaning of roads',
            '✓ Dust control at construction sites',
            '✓ Water spraying to reduce dust particles',
            '✓ Vegetative cover on open ground'
        ]
    },
    'Vehicle Count': {
        'priority': 'HIGH',
        'measures': [
            '✓ Promote public transportation (buses, metro)',
            '✓ Encourage carpooling and ride-sharing',
            '✓ Implement odd-even vehicle schemes',
            '✓ Work from home policies'
        ]
    },
    'Industrial Activity Index': {
        'priority': 'HIGH',
        'measures': [
            '✓ Enforce green manufacturing practices',
            '✓ Install scrubbers on industrial chimneys',
            '✓ Regular stack emission monitoring'
        ]
    },
    'NO2': {
        'priority': 'HIGH',
        'measures': [
            '✓ Control vehicle emissions at source',
            '✓ Improve fuel quality standards',
            '✓ Regular vehicle maintenance campaigns'
        ]
    }
}

# ===== LOAD & CACHE DATA & MODELS =====
@st.cache_resource
def load_data_and_train_models():
    """Load data and train models"""
    df = pd.read_csv('files/indian_aqi_realistic_2019_2024.csv')
    
    # Categorize AQI
    def categorize_aqi(aqi_value):
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
    
    df['AQI_Category'] = df['AQI'].apply(categorize_aqi)
    
    # Features
    features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 
                'Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 
                'Rainfall (mm)', 'Pressure (hPa)', 'Vehicle Count', 
                'Industrial Activity Index']
    
    X = df[features]
    y_category = df['AQI_Category']
    y_aqi = df['AQI']
    
    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Train regressor
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_aqi, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    rf_regressor.fit(X_train_reg, y_train_reg)
    
    return df, rf_classifier, rf_regressor, features

# ===== PREDICTION FUNCTION =====
def predict_aqi(rf_regressor, rf_classifier, features_list, feature_values):
    """Make AQI prediction"""
    input_data = pd.DataFrame([feature_values], columns=features_list)
    aqi_pred = rf_regressor.predict(input_data)[0]
    category_pred = rf_classifier.predict(input_data)[0]
    return aqi_pred, category_pred

# ===== CATEGORIZE AQI =====
def categorize_aqi(aqi_value):
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

# ===== MAIN APP =====
def main():
    # Initialize session state for city data loading
    if 'loaded_city_data' not in st.session_state:
        st.session_state.loaded_city_data = None
    
    # Header
    st.title("🌍 Air Quality Index (AQI) Prediction System")
    st.markdown("### 🤖 Powered by Random Forest Machine Learning")
    
    # Load data and models
    df, rf_classifier, rf_regressor, features = load_data_and_train_models()
    
    # Sidebar navigation
    st.sidebar.markdown("## 📋 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🔮 Predict AQI", "📊 Data Analysis", "📈 Top Recommendations", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        st.header("Welcome to AQI Prediction System")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Records", f"{len(df):,}")
        with col2:
            st.metric("🏙️ Cities Analyzed", df['City'].nunique())
        with col3:
            st.metric("📈 AQI Range", f"{df['AQI'].min():.0f} - {df['AQI'].max():.0f}")
        
        st.markdown("---")
        st.markdown("""
        ### What is this System?
        
        This AI-powered system uses **Random Forest Machine Learning** to:
        - 🎯 **Predict AQI categories** (Good → Severe)
        - 🔴 **Classify air quality** as SAFE or HAZARDOUS
        - ⚠️ **Show health effects** for each AQI level
        - 💡 **Suggest improvement measures** to reduce pollution
        
        ### 🎯 Three Ways to Use This App:
        
        #### 1️⃣ **Quick Scenarios** (No Data Entry)
        > Perfect if you don't have pollutant measurements
        - Select from 4 preset scenarios: Good, Moderate, Poor, Severe
        - Get instant predictions with one click
        - Great for understanding AQI impact levels
        
        #### 2️⃣ **Real City Data** (Most Practical)
        > Use actual data from our dataset
        - Select your city from dropdown
        - Auto-loads historical average pollutant values
        - No manual entry needed!
        - Works for 24 Indian cities
        
        #### 3️⃣ **Custom Parameters** (Advanced)
        > For detailed analysis
        - Use sliders to adjust values
        - Compare different scenarios
        - For researchers and analysts
        """)
        
        # Show category guide
        st.markdown("### 📊 AQI Categories at a Glance")
        
        categories_data = []
        for cat in ['Good', 'Satisfactory', 'Moderately Polluted', 'Poor', 'Very Poor', 'Severe']:
            info = HEALTH_EFFECTS_DB[cat]
            categories_data.append({
                'Category': f"{info['emoji']} {cat}",
                'Range': info['range'],
                'Status': info['status']
            })
        
        cat_df = pd.DataFrame(categories_data)
        st.table(cat_df)
    
    elif page == "🔮 Predict AQI":
        st.header("🔮 Predict AQI for Your City")
        
        # Quick scenarios
        st.markdown("### ⚡ Quick Scenarios")
        
        scenario_presets = {
            'Good Air Quality': {
                'pm25': 15.0, 'pm10': 30.0, 'no2': 20.0, 'co': 0.5, 'so2': 10.0, 'o3': 20.0,
                'temp': 25.0, 'humidity': 45.0, 'wind': 8.0, 'rainfall': 2.0, 'pressure': 1013.0,
                'vehicles': 50000.0, 'industrial': 2.5
            },
            'Moderate Pollution': {
                'pm25': 55.0, 'pm10': 100.0, 'no2': 45.0, 'co': 0.9, 'so2': 20.0, 'o3': 40.0,
                'temp': 28.0, 'humidity': 55.0, 'wind': 6.0, 'rainfall': 1.0, 'pressure': 1012.0,
                'vehicles': 100000.0, 'industrial': 5.0
            },
            'Poor Air Quality': {
                'pm25': 85.0, 'pm10': 150.0, 'no2': 65.0, 'co': 1.2, 'so2': 25.0, 'o3': 45.0,
                'temp': 28.0, 'humidity': 60.0, 'wind': 5.0, 'rainfall': 0.0, 'pressure': 1010.0,
                'vehicles': 150000.0, 'industrial': 7.5
            },
            'Severe Pollution': {
                'pm25': 180.0, 'pm10': 280.0, 'no2': 120.0, 'co': 2.5, 'so2': 50.0, 'o3': 80.0,
                'temp': 32.0, 'humidity': 70.0, 'wind': 3.0, 'rainfall': 0.0, 'pressure': 1008.0,
                'vehicles': 250000.0, 'industrial': 9.8
            }
        }
        
        selected_scenario = st.radio("Select a scenario:", list(scenario_presets.keys()), horizontal=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_preset = st.checkbox("Use preset values", value=True)
        
        with col2:
            city_name = st.text_input("City Name (optional):", value="Test City")
        
        # Get values from selected scenario or from input
        if use_preset:
            preset_values = scenario_presets[selected_scenario]
            st.info(f"📌 Using preset: **{selected_scenario}**")
        else:
            st.markdown("### 📊 Enter or Adjust Values")
        
        # Input columns with sliders for better UX (only show if not using loaded city data)
        if not st.session_state.loaded_city_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if use_preset:
                    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, preset_values['pm25'])
                    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, preset_values['pm10'])
                    no2 = st.slider("NO2 (ppb)", 0.0, 200.0, preset_values['no2'])
                else:
                    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, 50.0)
                    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 100.0)
                    no2 = st.slider("NO2 (ppb)", 0.0, 200.0, 40.0)
            
            with col2:
                if use_preset:
                    co = st.slider("CO (ppm)", 0.0, 5.0, preset_values['co'])
                    so2 = st.slider("SO2 (ppb)", 0.0, 100.0, preset_values['so2'])
                    o3 = st.slider("O3 (ppb)", 0.0, 200.0, preset_values['o3'])
                else:
                    co = st.slider("CO (ppm)", 0.0, 5.0, 1.0)
                    so2 = st.slider("SO2 (ppb)", 0.0, 100.0, 20.0)
                    o3 = st.slider("O3 (ppb)", 0.0, 200.0, 40.0)
            
            with col3:
                if use_preset:
                    temp = st.slider("Temperature (°C)", -40.0, 60.0, preset_values['temp'])
                    humidity = st.slider("Humidity (%)", 0.0, 100.0, preset_values['humidity'])
                    wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, preset_values['wind'])
                else:
                    temp = st.slider("Temperature (°C)", -40.0, 60.0, 28.0)
                    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
                    wind = st.slider("Wind Speed (km/h)", 0.0, 50.0, 5.0)
            
            with col4:
                if use_preset:
                    rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, preset_values['rainfall'])
                    pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, preset_values['pressure'])
                    vehicles = st.slider("Vehicle Count", 10000.0, 500000.0, preset_values['vehicles'])
                    industrial = st.slider("Industrial Activity", 0.0, 10.0, preset_values['industrial'])
                else:
                    rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 0.0)
                    pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0)
                    vehicles = st.slider("Vehicle Count", 10000.0, 500000.0, 100000.0)
                    industrial = st.slider("Industrial Activity", 0.0, 10.0, 5.0)
        
        st.markdown("---")
        st.markdown("### 🏙️ OR Use Real City Data from Dataset")
        
        # Get unique cities from dataset
        cities_in_data = sorted(df['City'].unique().tolist())
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_city_data = st.selectbox("Select a city:", cities_in_data)
        
        with col2:
            if st.button("📍 Load Data", use_container_width=True):
                # Get MEDIAN/AVERAGE values for the selected city (not just first row)
                city_data_all = df[df['City'] == selected_city_data]
                city_avg = city_data_all.median(numeric_only=True)  # Use median to avoid outliers
                
                # Get the actual median AQI for display
                median_aqi = city_data_all['AQI'].median()
                
                st.session_state.loaded_city_data = {
                    'pm25': float(city_avg['PM2.5']),
                    'pm10': float(city_avg['PM10']),
                    'no2': float(city_avg['NO2']),
                    'co': float(city_avg['CO']),
                    'so2': float(city_avg['SO2']),
                    'o3': float(city_avg['O3']),
                    'temp': float(city_avg['Temperature (°C)']),
                    'humidity': float(city_avg['Humidity (%)']),
                    'wind': float(city_avg['Wind Speed (km/h)']),
                    'rainfall': float(city_avg['Rainfall (mm)']),
                    'pressure': float(city_avg['Pressure (hPa)']),
                    'vehicles': float(city_avg['Vehicle Count']),
                    'industrial': float(city_avg['Industrial Activity Index']),
                    'city_name': selected_city_data,
                    'median_aqi': int(median_aqi)
                }
                st.success(f"✓ Loaded real data for {selected_city_data} (Median AQI: {int(median_aqi)})")
                st.rerun()
        
        st.markdown("---")
        
        # Use loaded city data if available, otherwise use preset values
        if st.session_state.loaded_city_data:
            city_data = st.session_state.loaded_city_data
            pm25 = city_data['pm25']
            pm10 = city_data['pm10']
            no2 = city_data['no2']
            co = city_data['co']
            so2 = city_data['so2']
            o3 = city_data['o3']
            temp = city_data['temp']
            humidity = city_data['humidity']
            wind = city_data['wind']
            rainfall = city_data['rainfall']
            pressure = city_data['pressure']
            vehicles = city_data['vehicles']
            industrial = city_data['industrial']
            city_name = city_data['city_name']
            st.info(f"📌 Using data for {city_name} (loaded from dataset)")
        
        # Make prediction
        if st.button("🔮 Predict AQI", use_container_width=True):
            # If city data is loaded, use the actual AQI from dataset instead of predicting
            if st.session_state.loaded_city_data:
                aqi_value = float(st.session_state.loaded_city_data['median_aqi'])
                # Categorize the AQI value
                if aqi_value <= 50:
                    aqi_category = 'Good'
                elif aqi_value <= 100:
                    aqi_category = 'Satisfactory'
                elif aqi_value <= 150:
                    aqi_category = 'Moderately Polluted'
                elif aqi_value <= 200:
                    aqi_category = 'Poor'
                elif aqi_value <= 300:
                    aqi_category = 'Very Poor'
                else:
                    aqi_category = 'Severe'
            else:
                # Use prediction for custom values
                feature_values = {
                    'PM2.5': pm25,
                    'PM10': pm10,
                    'NO2': no2,
                    'CO': co,
                    'SO2': so2,
                    'O3': o3,
                    'Temperature (°C)': temp,
                    'Humidity (%)': humidity,
                    'Wind Speed (km/h)': wind,
                    'Rainfall (mm)': rainfall,
                    'Pressure (hPa)': pressure,
                    'Vehicle Count': vehicles,
                    'Industrial Activity Index': industrial
                }
                
                aqi_value, aqi_category = predict_aqi(rf_regressor, rf_classifier, features, list(feature_values.values()))
            
            # Display prediction
            st.markdown("---")
            st.markdown("### 📈 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            effect_info = HEALTH_EFFECTS_DB[aqi_category]
            
            with col1:
                st.metric("AQI Value", f"{aqi_value:.1f}", delta="Air Quality")
            with col2:
                st.metric("Category", f"{effect_info['emoji']} {aqi_category}")
            with col3:
                st.metric("Status", effect_info['status'])
            
            # Health effects
            st.markdown(f"### {effect_info['emoji']} Health Impact Warnings")
            
            if effect_info['severity'] <= 2:
                alert_type = "success"
            elif effect_info['severity'] <= 3:
                alert_type = "warning"
            else:
                alert_type = "error"
            
            st.markdown(f"**Status:** {effect_info['status']}")
            
            for effect in effect_info['health_effects']:
                st.write(effect)
            
            st.warning(f"**Precautions:** {effect_info['precautions']}")
            
            # Recommendations
            st.markdown(f"### 💡 Recommended Improvements for {city_name}")
            
            feature_importance_list = [
                ('PM2.5', 0.0770),
                ('PM10', 0.0773),
                ('Vehicle Count', 0.0761),
                ('Industrial Activity Index', 0.0798),
                ('NO2', 0.0786)
            ]
            
            for idx, (feature_name, _) in enumerate(feature_importance_list[:5], 1):
                if feature_name in IMPROVEMENT_MEASURES:
                    measures = IMPROVEMENT_MEASURES[feature_name]
                    st.markdown(f"**{idx}. {feature_name} - Priority: {measures['priority']}**")
                    for measure in measures['measures'][:3]:
                        st.write(measure)
    
    elif page == "📊 Data Analysis":
        st.header("📊 Data Analysis & Visualization")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cities", df['City'].nunique())
        with col2:
            st.metric("Average AQI", f"{df['AQI'].mean():.1f}")
        with col3:
            st.metric("Max AQI", f"{df['AQI'].max():.0f}")
        
        st.markdown("---")
        
        # AQI Category Distribution
        st.markdown("### AQI Category Distribution")
        category_counts = df['AQI_Category'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad', '#c0392b']
        ax.bar(category_counts.index, category_counts.values, color=colors[:len(category_counts)])
        ax.set_title('AQI Category Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Records')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Top Pollutants
        st.markdown("### Top Pollutants Distribution")
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, pollutant in enumerate(pollutants):
            axes[idx].hist(df[pollutant], bins=30, color='steelblue', edgecolor='black')
            axes[idx].set_title(f'{pollutant} Distribution')
            axes[idx].set_xlabel('Concentration')
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # City-wise analysis
        st.markdown("### City-wise Average AQI")
        city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(city_aqi.index, city_aqi.values, color='coral')
        ax.set_title('Top 10 Most Polluted Cities (Average AQI)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Average AQI')
        st.pyplot(fig)
    
    elif page == "📈 Top Recommendations":
        st.header("📈 Top Recommendations to Reduce AQI")
        
        recommendations = [
            {
                'rank': 1,
                'priority': 'CRITICAL',
                'pollutant': 'PM2.5 (Fine Particulate Matter)',
                'impact': '7.70%',
                'measures': [
                    'Install HEPA air purifiers in homes and offices',
                    'Enforce industrial emission standards strictly',
                    'Promote electric vehicles and reduce diesel use',
                    'Control construction dust with barriers',
                    'Ban agricultural stubble burning'
                ]
            },
            {
                'rank': 2,
                'priority': 'CRITICAL',
                'pollutant': 'PM10 (Coarse Particulate Matter)',
                'impact': '7.73%',
                'measures': [
                    'Regular street sweeping and wet cleaning',
                    'Dust control at construction sites',
                    'Water spraying to reduce dust',
                    'Vegetative cover in open areas'
                ]
            },
            {
                'rank': 3,
                'priority': 'HIGH',
                'pollutant': 'Vehicle Emissions',
                'impact': '7.61%',
                'measures': [
                    'Expand public transportation',
                    'Encourage carpooling and ride-sharing',
                    'Implement odd-even vehicle schemes',
                    'Support work-from-home policies'
                ]
            },
            {
                'rank': 4,
                'priority': 'HIGH',
                'pollutant': 'Industrial Activity',
                'impact': '7.98%',
                'measures': [
                    'Enforce green manufacturing practices',
                    'Install pollution control equipment',
                    'Relocate polluting industries',
                    'Monitor emissions regularly'
                ]
            },
            {
                'rank': 5,
                'priority': 'HIGH',
                'pollutant': 'Nitrogen Dioxide (NO2)',
                'impact': '7.86%',
                'measures': [
                    'Control vehicle emissions',
                    'Improve fuel quality standards',
                    'Mandate vehicle maintenance',
                    'Promote renewable energy'
                ]
            }
        ]
        
        for rec in recommendations:
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.markdown(f"### #{rec['rank']}")
                st.markdown(f"**{rec['priority']}**")
                st.markdown(f"*Impact: {rec['impact']}*")
            
            with col2:
                st.markdown(f"### {rec['pollutant']}")
                for measure in rec['measures']:
                    st.write(f"✓ {measure}")
            
            st.markdown("---")
    
    elif page == "ℹ️ About":
        st.header("ℹ️ About this System")
        
        st.markdown("""
        ### 🎓 Project Overview
        
        This **Random Forest Machine Learning** system predicts Air Quality Index (AQI) and provides:
        
        - **AQI Category Prediction** - Classifies air quality from Safe to Hazardous
        - **Health Impact Assessment** - Shows specific health effects for each category
        - **Improvement Measures** - Suggests actionable solutions based on feature importance
        - **Data-Driven Insights** - Analyzes 10,000+ records from 24 Indian cities (2019-2024)
        
        ### 🚀 Key Features
        
        ✅ Real-time AQI prediction for any city
        ✅ Comprehensive health effect warnings
        ✅ Evidence-based improvement recommendations
        ✅ Interactive data visualizations
        ✅ Historical trend analysis
        
        ### 🔬 Technical Details
        
        - **Model Type:** Random Forest Classifier & Regressor
        - **Features Used:** 13 environmental and pollution indicators
        - **Data Points:** 10,000+ records
        - **Cities Covered:** 24 Indian cities
        - **Time Period:** 2019-2024
        
        ### 📊 AQI Scale
        
        | Category | Range | Status |
        |----------|-------|--------|
        | 🟢 Good | 0-50 | SAFE |
        | 🟡 Satisfactory | 51-100 | SAFE |
        | 🟠 Moderately Polluted | 101-150 | MODERATE HAZARD |
        | 🔴 Poor | 151-200 | HAZARDOUS |
        | 🟣 Very Poor | 201-300 | HAZARDOUS |
        | 💀 Severe | 301+ | HAZARDOUS |
        
        ### 👨‍💼 Use Cases
        
        - **Public Health:** Early warning systems for hazardous air quality
        - **Policy Making:** Data-driven recommendations for pollution control
        - **Urban Planning:** Identify key pollution sources
        - **Research:** Study air quality patterns and trends
        - **Education:** Learn about air pollution and health impacts
        
        ### 📞 Support
        
        For more information, check the project documentation or consult the data analysis section.
        """)

if __name__ == "__main__":
    main()
