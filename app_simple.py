"""
AQI Prediction System - Simplified Streamlit App
Works with trained models directly
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide"
)

# ===== STYLING =====
st.markdown("""
<style>
    .main-title { font-size: 2.5em; color: #1f77b4; font-weight: bold; }
    .section-title { font-size: 1.8em; color: #2c92a5; font-weight: bold; border-bottom: 2px solid #2c92a5; padding-bottom: 10px; }
    .aqi-good { background-color: #d4edda; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .aqi-moderate { background-color: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .aqi-poor { background-color: #f8d7da; padding: 15px; border-radius: 8px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ===== LOAD MODELS =====
@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        rf_model = joblib.load('models/random_forest_model.pkl')
        gb_model = joblib.load('models/gradient_boosting_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        location_stats = joblib.load('models/location_stats.pkl')
        city_mapping = joblib.load('models/city_location_mapping.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        return {
            'rf': rf_model,
            'gb': gb_model,
            'scaler': scaler,
            'location_stats': location_stats,
            'city_mapping': city_mapping,
            'label_encoders': label_encoders,
            'features': feature_names
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    """Load cleaned dataset"""
    try:
        df = pd.read_csv('files/station_hour_cleaned.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ===== AQI PREDICTION MODEL =====
def predict_aqi(models, year, month, pm25, pm10, station_name, city_name):
    """Predict AQI using ensemble of RF and GB"""
    if not models:
        return None
    
    try:
        le_station = models['label_encoders']['station']
        le_city = models['label_encoders']['city']
        
        # Encode categorical
        station_enc = le_station.transform([station_name])[0]
        city_enc = le_city.transform([city_name])[0]
        
        # Create feature vector
        features = [year, month, pm25, pm10, station_enc, city_enc]
        X = np.array(features).reshape(1, -1)
        
        # Scale
        X_scaled = models['scaler'].transform(X)
        
        # Predict
        rf_pred = models['rf'].predict(X_scaled)[0]
        gb_pred = models['gb'].predict(X_scaled)[0]
        
        # Ensemble
        ensemble_pred = (rf_pred + gb_pred) / 2
        ensemble_pred = max(0, ensemble_pred)  # No negative AQI
        
        return {
            'rf': round(rf_pred, 1),
            'gb': round(gb_pred, 1),
            'ensemble': round(ensemble_pred, 1)
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ===== AQI CATEGORY =====
def get_aqi_category(aqi):
    """Get AQI category and recommendations"""
    if aqi <= 50:
        return "Good", "🟢", "SAFE", "#d4edda"
    elif aqi <= 100:
        return "Satisfactory", "🟡", "SAFE", "#fff3cd"
    elif aqi <= 150:
        return "Moderate", "🟠", "CAUTION", "#ffe0b2"
    elif aqi <= 200:
        return "Poor", "🔴", "HAZARDOUS", "#ffcccc"
    elif aqi <= 300:
        return "Very Poor", "🟣", "CRITICAL", "#e1bee7"
    else:
        return "Severe", "💀", "EMERGENCY", "#721c24"

# ===== HEALTH EFFECTS =====
HEALTH_EFFECTS = {
    'Good': ['✓ No adverse health effects', '✓ All activities safe', '✓ Excellent air quality'],
    'Satisfactory': ['⚠ Minimal risk', '⚠ Occasional symptoms', '⚠ Sensitive groups should limit'],
    'Moderate': ['⛔ Respiratory discomfort', '⛔ Asthma attacks possible', '⛔ Avoid strenuous activities'],
    'Poor': ['🚨 Severe respiratory illness', '🚨 Hospital admissions likely', '🚨 STAY INDOORS'],
    'Very Poor': ['🚨 Life-threatening', '🚨 Emergency protocols', '🚨 Use N100 masks'],
    'Severe': ['💀 Mass casualties risk', '💀 Total lockdown', '💀 Evacuation alert']
}

PREVENTION = {
    'Good': ['Enjoy outdoor activities', 'No restrictions needed', 'Open windows for ventilation'],
    'Satisfactory': ['Limit prolonged outdoor exposure', 'Sensitive groups use masks', 'Monitor forecast'],
    'Moderate': ['Avoid outdoor activities', 'Use N95/N99 masks', 'Increase air filtration'],
    'Poor': ['STAY INDOORS', 'Use N95 masks if outside', 'Keep medical help ready'],
    'Very Poor': ['CRITICAL - Stay indoors', 'Use N100 masks only', 'Hospital on alert'],
    'Severe': ['LOCKDOWN - NO outdoor movement', 'Emergency services active', 'Evacuation ready']
}

# ===== MAIN APP =====
def main():
    st.markdown("<h1 class='main-title'>🌍 AQI Prediction & Prevention System</h1>", unsafe_allow_html=True)
    st.write("Predict air quality and get health recommendations")
    
    # Load resources
    models = load_models()
    df = load_data()
    
    if not models or df is None:
        st.error("Failed to load models or data. Please check the models/ folder.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictions", "🗺️ Analysis", "⚠️ Prevention", "📊 Dashboard"])
    
    # ===== TAB 1: PREDICTIONS =====
    with tab1:
        st.markdown("<h2 class='section-title'>Predict Future AQI</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cities = sorted(models['city_mapping'].keys())
            selected_city = st.selectbox("🏙️ City", cities)
        
        with col2:
            stations = models['city_mapping'].get(selected_city, [])
            selected_station = st.selectbox("📍 Station", stations)
        
        with col3:
            st.write("")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pred_year = st.number_input("Year", 2024, 2026, 2025)
        
        with col2:
            pred_month = st.number_input("Month", 1, 12, 6)
        
        with col3:
            pm25 = st.number_input("PM2.5", 0.0, 500.0, 50.0)
        
        with col4:
            pm10 = st.number_input("PM10", 0.0, 500.0, 100.0)
        
        if st.button("🔍 Predict AQI", use_container_width=True):
            pred = predict_aqi(models, pred_year, pred_month, pm25, pm10, selected_station, selected_city)
            
            if pred:
                aqi = pred['ensemble']
                category, emoji, status, color = get_aqi_category(aqi)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Predicted AQI", f"{aqi:.0f}")
                
                with col2:
                    st.metric("Category", category)
                
                with col3:
                    st.metric("Status", status)
                
                with col4:
                    st.metric("Month/Year", f"{pred_month:02d}/{pred_year}")
                
                # AQI Gauge
                st.markdown(f"#### {emoji} {category.upper()} ({status})")
                
                fig, ax = plt.subplots(figsize=(12, 2))
                categories = [0, 50, 100, 150, 200, 300, 350]
                colors_bar = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1', '#721c24', '#000000']
                
                for i in range(len(categories)-1):
                    width = categories[i+1] - categories[i]
                    ax.barh(0, width, left=categories[i], height=0.3, color=colors_bar[i])
                
                ax.plot([aqi, aqi], [-0.2, 0.2], 'k-', linewidth=4)
                ax.set_xlim(0, 350)
                ax.set_ylim(-1, 1)
                ax.set_xlabel('AQI Index', fontweight='bold')
                ax.set_xticks(categories)
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig, use_container_width=True)
                
                # Predictions breakdown
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Model Predictions:**")
                    st.info(f"Random Forest: {pred['rf']:.1f} AQI\nGradient Boosting: {pred['gb']:.1f} AQI\nEnsemble: {aqi:.1f} AQI")
                
                with col2:
                    # Historical stats
                    location_info = models['location_stats'].get(selected_station, {})
                    st.markdown("**Historical Data:**")
                    st.info(f"Avg AQI: {location_info.get('avg_aqi', 'N/A'):.0f}\nStd Dev: {location_info.get('std_aqi', 'N/A'):.0f}")
                
                # Health effects
                st.markdown("#### 🏥 Health Effects at This Level")
                st.markdown(f"<div style='background-color: {color}; padding: 15px; border-radius: 8px;'>", unsafe_allow_html=True)
                for effect in HEALTH_EFFECTS.get(category, []):
                    st.markdown(effect)
                st.markdown("</div>", unsafe_allow_html=True)
    
    # ===== TAB 2: SPATIAL ANALYSIS =====
    with tab2:
        st.markdown("<h2 class='section-title'>Spatial AQI Analysis</h2>", unsafe_allow_html=True)
        
        analysis_city = st.selectbox("🏙️ Select City", sorted(models['city_mapping'].keys()))
        analysis_type = st.radio("📊 View", ["High-Risk Locations", "Safe Locations", "All Stations"])
        
        all_stats = models['location_stats']
        
        if analysis_type == "High-Risk Locations":
            high_risk = {s: stats for s, stats in all_stats.items() 
                        if stats['city'] == analysis_city and stats['avg_aqi'] > 150}
            
            if high_risk:
                st.warning(f"⚠️ {len(high_risk)} High-Risk Locations")
                for station, stats in sorted(high_risk.items(), key=lambda x: x[1]['avg_aqi'], reverse=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{station}**")
                    with col2:
                        st.metric("Avg AQI", f"{stats['avg_aqi']:.0f}")
            else:
                st.success("✅ No high-risk locations!")
        
        elif analysis_type == "Safe Locations":
            safe = {s: stats for s, stats in all_stats.items() 
                   if stats['city'] == analysis_city and stats['avg_aqi'] < 100}
            
            if safe:
                st.success(f"✅ {len(safe)} Safe Locations")
                for station, stats in sorted(safe.items(), key=lambda x: x[1]['avg_aqi']):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{station}**")
                    with col2:
                        st.metric("Avg AQI", f"{stats['avg_aqi']:.0f}")
            else:
                st.info("No very safe locations")
        
        else:  # All stations
            city_stats = {s: stats for s, stats in all_stats.items() if stats['city'] == analysis_city}
            
            if city_stats:
                data = []
                for station, stats in city_stats.items():
                    data.append({
                        'Station': station,
                        'Avg AQI': stats['avg_aqi'],
                        'Std Dev': stats['std_aqi'],
                        'Records': stats['records']
                    })
                
                stats_df = pd.DataFrame(data).sort_values('Avg AQI', ascending=False)
                st.dataframe(stats_df, use_container_width=True)
                
                # Chart
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(stats_df['Station'], stats_df['Avg AQI'])
                ax.set_xlabel('Average AQI')
                ax.set_title(f'Stations in {analysis_city}')
                st.pyplot(fig, use_container_width=True)
    
    # ===== TAB 3: PREVENTION GUIDE =====
    with tab3:
        st.markdown("<h2 class='section-title'>Prevention & Health Guide</h2>", unsafe_allow_html=True)
        
        aqi_level = st.slider("Select AQI Level", 0, 500, 100)
        category, emoji, status, color = get_aqi_category(aqi_level)
        
        st.markdown(f"#### {emoji} {category.upper()} ({status})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Health Effects:**")
            for effect in HEALTH_EFFECTS.get(category, []):
                st.markdown(effect)
        
        with col2:
            st.markdown("**Prevention Steps:**")
            for step in PREVENTION.get(category, []):
                st.markdown(f"✓ {step}")
    
    # ===== TAB 4: DASHBOARD =====
    with tab4:
        st.markdown("<h2 class='section-title'>AQI Statistics Dashboard</h2>", unsafe_allow_html=True)
        
        all_aqi = [s['avg_aqi'] for s in models['location_stats'].values()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stations", len(models['location_stats']))
        with col2:
            st.metric("Avg AQI", f"{np.mean(all_aqi):.1f}")
        with col3:
            st.metric("Median AQI", f"{np.median(all_aqi):.1f}")
        with col4:
            st.metric("Max AQI", f"{np.max(all_aqi):.1f}")
        
        # Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            ax.hist(all_aqi, bins=25, color='steelblue', edgecolor='black')
            ax.set_xlabel('AQI Index')
            ax.set_ylabel('Count')
            ax.set_title('AQI Distribution')
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            categories_count = {'Good': 0, 'Satisfactory': 0, 'Moderate': 0, 'Poor': 0, 'Very Poor': 0, 'Severe': 0}
            for aqi in all_aqi:
                cat, _, _, _ = get_aqi_category(aqi)
                if cat in categories_count:
                    categories_count[cat] += 1
            
            fig, ax = plt.subplots()
            ax.pie([v for v in categories_count.values() if v > 0], 
                   labels=[k for k, v in categories_count.items() if v > 0],
                   autopct='%1.1f%%')
            ax.set_title('AQI Category Distribution')
            st.pyplot(fig, use_container_width=True)

if __name__ == "__main__":
    main()
