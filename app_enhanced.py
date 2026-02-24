"""
Enhanced AQI Prediction System with Spatial-Temporal Analysis
Frontend: Streamlit Application
Features: Location-based predictions, prevention guides, health recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from aqi_spatial_temporal_model import SpatialTemporalAQIPredictor
from aqi_prevention_guide import AQIPreventionGuide

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="AQI Spatial-Temporal Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM STYLING =====
st.markdown("""
<style>
    .main-header { font-size: 2.5em; color: #1f77b4; font-weight: bold; margin-bottom: 20px; }
    .section-header { font-size: 1.8em; color: #2c92a5; font-weight: bold; margin-top: 30px; margin-bottom: 15px; border-bottom: 3px solid #2c92a5; padding-bottom: 10px; }
    .metric-box { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; border-left: 5px solid #1f77b4; }
    .safe { background-color: #d4edda; border-left-color: #28a745; }
    .warning { background-color: #fff3cd; border-left-color: #ffc107; }
    .danger { background-color: #f8d7da; border-left-color: #dc3545; }
    .severe { background-color: #721c24; color: white; border-left-color: #721c24; }
    .info-text { font-size: 14px; padding: 10px; border-radius: 5px; line-height: 1.6; }
    .location-card { background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #17a2b8; }
    .prevention-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #6c757d; }
    .aqi-good { background-color: #d4edda; color: #155724; }
    .aqi-satisfactory { background-color: #fff3cd; color: #856404; }
    .aqi-moderate { background-color: #ffe0b2; color: #e65100; }
    .aqi-poor { background-color: #ffcccc; color: #b71c1c; }
    .aqi-very-poor { background-color: #e1bee7; color: #4a148c; }
    .aqi-severe { background-color: #721c24; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ===== INITIALIZE SESSION STATE =====
@st.cache_resource
def load_predictor():
    """Load and train the spatial-temporal predictor"""
    predictor = SpatialTemporalAQIPredictor(data_path='files/station_hour_cleaned.csv')
    predictor.load_and_prepare_data()
    metrics = predictor.train_models()
    return predictor

# ===== MAIN APP =====
def main():
    # Title
    st.markdown("<div class='main-header'>🌍 AQI Spatial-Temporal Prediction System</div>", unsafe_allow_html=True)
    st.write("Advanced prediction of Air Quality Index by location and time with prevention recommendations")
    
    # Load predictor
    with st.spinner("Loading prediction models..."):
        predictor = load_predictor()
    
    # Create navigation tabs
    tabs = st.tabs([
        "🔮 Future AQI Predictions",
        "🗺️ Spatial Analysis",
        "⚠️ Prevention Guide",
        "📊 Data Dashboard",
        "🏥 Health Impacts"
    ])
    
    # ===== TAB 1: FUTURE AQI PREDICTIONS =====
    with tabs[0]:
        st.markdown("<div class='section-header'>📈 Predict Future AQI by Location and Time</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cities = predictor.get_all_cities()
            selected_city = st.selectbox("🏙️ Select City", cities, key="city_select_pred")
        
        with col2:
            stations = predictor.get_city_stations(selected_city)
            selected_station = st.selectbox("📍 Select Monitoring Station", stations, key="station_select_pred")
        
        with col3:
            st.write("")  # Spacing
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prediction_month = st.selectbox("📅 Month", list(range(1, 13)), format_func=lambda x: f"Month {x:02d}", key="month_select")
        
        with col2:
            prediction_year = st.slider("📆 Year", 2024, 2026, 2025, key="year_select")
        
        with col3:
            if st.button("🔍 Predict AQI", key="predict_btn", use_container_width=True):
                with st.spinner("Generating predictions..."):
                    # Get prediction for selected station
                    pred = predictor.predict_spatial_temporal_aqi(selected_station, prediction_year, prediction_month)
                    
                    if pred:
                        # Display prediction
                        col1, col2, col3, col4 = st.columns(4)
                        
                        aqi_value = pred['predicted_aqi']
                        aqi_category = AQIPreventionGuide.get_aqi_category(aqi_value)
                        health_info = AQIPreventionGuide.get_health_effects(aqi_category)
                        
                        with col1:
                            st.metric("Predicted AQI", f"{aqi_value:.0f}", f"Ensemble Avg")
                        
                        with col2:
                            st.metric("Category", aqi_category, f"{health_info['emoji']}")
                        
                        with col3:
                            st.metric("Confidence", pred['confidence'], f"±{pred['historical_std']:.0f}")
                        
                        with col4:
                            st.metric("Historical Avg", f"{pred['historical_avg']:.0f}", "Same Station")
                        
                        # AQI Gauge
                        fig, ax = plt.subplots(figsize=(10, 2))
                        aqi_ranges = [0, 50, 100, 150, 200, 300]
                        colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1', '#721c24']
                        
                        for i in range(len(aqi_ranges)-1):
                            ax.barh(0, aqi_ranges[i+1]-aqi_ranges[i], left=aqi_ranges[i], height=0.5, color=colors[i], edgecolor='black')
                        
                        ax.plot([aqi_value, aqi_value], [-0.3, 0.3], 'k-', linewidth=3, marker='v', markersize=10)
                        ax.set_xlim(0, 350)
                        ax.set_ylim(-1, 1)
                        ax.set_xlabel('AQI Index', fontsize=12, fontweight='bold')
                        ax.set_xticks(aqi_ranges)
                        ax.set_yticks([])
                        ax.spines['left'].set_visible(False)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        st.pyplot(fig, use_container_width=True)
                        
                        # Detailed Prediction Info
                        st.markdown("#### 📊 Detailed Prediction Information")
                        
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            st.markdown(f"""
                            **Station Location:** {pred['station']}
                            **City:** {pred['city']}
                            **Prediction Date:** {prediction_month:02d}/{prediction_year}
                            
                            **Model Predictions:**
                            - Random Forest: {pred['rf_prediction']:.1f} AQI
                            - Gradient Boosting: {pred['gb_prediction']:.1f} AQI
                            - **Ensemble Average: {pred['predicted_aqi']:.1f} AQI**
                            """)
                        
                        with info_col2:
                            st.markdown(f"""
                            **Historical Statistics:**
                            - Average AQI: {pred['historical_avg']:.1f}
                            - Std Deviation: {pred['historical_std']:.1f}
                            - Variation Range: ±{pred['historical_std']:.1f}
                            
                            **Prediction Quality:**
                            - Confidence Level: {pred['confidence']}
                            - Prediction Reliability: High
                            """)
                        
                        # Health Effects
                        st.markdown("#### 🏥 Health Effects at This AQI Level")
                        st.markdown(f"<div class='metric-box {aqi_category.lower().replace(' ', '-')}'>", unsafe_allow_html=True)
                        for effect in health_info['health_effects']:
                            st.markdown(effect)
                        st.markdown("</div>", unsafe_allow_html=True)
        
        # Predictions for all stations in city
        st.markdown("<div class='section-header'>🗺️ Predictions for All Stations in {}</div>".format(selected_city), unsafe_allow_html=True)
        
        if st.button("Show All Stations Predictions", key="show_all_stations"):
            with st.spinner("Predicting for all stations..."):
                city_predictions = predictor.predict_city_aqi(selected_city, prediction_year, prediction_month)
                
                if not city_predictions.empty:
                    # Sort by AQI
                    city_predictions = city_predictions.sort_values('predicted_aqi', ascending=False)
                    
                    # Display as interactive table
                    st.dataframe(
                        city_predictions[[
                            'station', 'predicted_aqi', 'rf_prediction', 
                            'gb_prediction', 'historical_avg', 'confidence'
                        ]].rename(columns={
                            'station': 'Station Name',
                            'predicted_aqi': 'Predicted AQI',
                            'rf_prediction': 'RF Model',
                            'gb_prediction': 'GB Model',
                            'historical_avg': 'Historical Avg',
                            'confidence': 'Confidence'
                        }),
                        use_container_width=True
                    )
                    
                    # Chart: Station-wise predictions
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.barh(
                        city_predictions['station'],
                        city_predictions['predicted_aqi'],
                        color=['#28a745' if x <= 50 else '#ffc107' if x <= 100 else '#fd7e14' if x <= 150 
                               else '#dc3545' if x <= 200 else '#6f42c1' if x <= 300 else '#721c24' 
                               for x in city_predictions['predicted_aqi']]
                    )
                    ax.axvline(x=100, color='orange', linestyle='--', linewidth=2, label='Safety Threshold')
                    ax.axvline(x=200, color='red', linestyle='--', linewidth=2, label='Hazard Threshold')
                    ax.set_xlabel('AQI Index', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Station', fontsize=12, fontweight='bold')
                    ax.set_title(f'AQI Predictions for {selected_city} - {prediction_month:02d}/{prediction_year}', 
                                fontsize=14, fontweight='bold')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
    
    # ===== TAB 2: SPATIAL ANALYSIS =====
    with tabs[1]:
        st.markdown("<div class='section-header'>🗺️ Spatial AQI Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            spatial_city = st.selectbox("🏙️ Select City", predictor.get_all_cities(), key="spatial_city")
        
        with col2:
            analysis_type = st.radio("📊 Analysis Type", ["High-Risk Locations", "Safe Locations", "All Locations"])
        
        if st.button("🔍 Analyze Spatial Distribution", use_container_width=True, key="spatial_btn"):
            if analysis_type == "High-Risk Locations":
                high_risk = predictor.get_high_risk_locations(spatial_city, threshold=150)
                
                if not high_risk.empty:
                    st.warning(f"⚠️ {len(high_risk)} High-Risk Locations Identified in {spatial_city}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.dataframe(
                            high_risk.rename(columns={
                                'station': 'Station Name',
                                'city': 'City',
                                'avg_aqi': 'Average AQI',
                                'risk_level': 'Risk Level',
                                'recommendation': 'Recommendation'
                            }),
                            use_container_width=True
                        )
                    
                    with col2:
                        # Risk distribution
                        risk_counts = high_risk['risk_level'].value_counts()
                        fig, ax = plt.subplots(figsize=(6, 6))
                        colors = ['#721c24', '#dc3545']
                        ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
                        ax.set_title('Risk Distribution')
                        st.pyplot(fig, use_container_width=True)
                    
                    # Location recommendations
                    st.markdown("#### 🚫 Locations to Avoid")
                    location_recs = AQIPreventionGuide.get_location_recommendations(spatial_city)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**High-Risk Areas:**")
                        for area in location_recs.get('high_risk_areas', []):
                            st.markdown(f"- ❌ {area}")
                    
                    with col2:
                        st.markdown("**Safer Areas:**")
                        for area in location_recs.get('safer_areas', []):
                            st.markdown(f"- ✅ {area}")
                else:
                    st.success(f"✅ No high-risk locations in {spatial_city}")
            
            elif analysis_type == "Safe Locations":
                safe = predictor.get_safe_locations(spatial_city, threshold=100)
                
                if not safe.empty:
                    st.success(f"✅ {len(safe)} Safe Locations Identified in {spatial_city}")
                    st.dataframe(
                        safe.rename(columns={
                            'station': 'Station Name',
                            'city': 'City',
                            'avg_aqi': 'Average AQI',
                            'air_quality': 'Air Quality',
                            'recommendation': 'Recommendation'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info(f"No very safe locations in {spatial_city}")
            
            else:  # All Locations
                st.markdown(f"#### All Monitoring Stations in {spatial_city}")
                all_stations = predictor.get_city_stations(spatial_city)
                
                stats_data = []
                for station in all_stations:
                    stats = predictor.location_stats.get(station, {})
                    stats_data.append({
                        'Station': station,
                        'Avg AQI': stats.get('avg_aqi', 0),
                        'Std Dev': stats.get('std_aqi', 0),
                        'Min': stats.get('min_aqi', 0),
                        'Max': stats.get('max_aqi', 0),
                        'Avg PM2.5': stats.get('avg_pm25', 0),
                        'Avg PM10': stats.get('avg_pm10', 0)
                    })
                
                stats_df = pd.DataFrame(stats_data).sort_values('Avg AQI', ascending=False)
                st.dataframe(stats_df, use_container_width=True)
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # AQI distribution
                ax1.barh(stats_df['Station'], stats_df['Avg AQI'], 
                        color=['#28a745' if x <= 50 else '#ffc107' if x <= 100 else '#fd7e14' 
                               if x <= 150 else '#dc3545' if x <= 200 else '#6f42c1' if x <= 300 
                               else '#721c24' for x in stats_df['Avg AQI']])
                ax1.set_xlabel('Average AQI')
                ax1.set_title(f'Average AQI by Station in {spatial_city}')
                
                # PM2.5 vs PM10
                ax2.scatter(stats_df['Avg PM2.5'], stats_df['Avg PM10'], s=100, alpha=0.6)
                for i, station in enumerate(stats_df['Station']):
                    ax2.annotate(station.split(',')[0][:20], 
                               (stats_df['Avg PM2.5'].iloc[i], stats_df['Avg PM10'].iloc[i]),
                               fontsize=8)
                ax2.set_xlabel('Average PM2.5')
                ax2.set_ylabel('Average PM10')
                ax2.set_title('PM2.5 vs PM10 Distribution')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
    
    # ===== TAB 3: PREVENTION GUIDE =====
    with tabs[2]:
        st.markdown("<div class='section-header'>⚠️ Prevention & Health Protection Guide</div>", unsafe_allow_html=True)
        
        guide = AQIPreventionGuide()
        
        # Select AQI level
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_aqi = st.slider("🎯 Select AQI Level to View Prevention Measures", 0, 500, 100)
        
        aqi_category = guide.get_aqi_category(selected_aqi)
        health_info = guide.get_health_effects(aqi_category)
        
        # Display category badge
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"#### {health_info['emoji']} Current AQI Level: {aqi_category}")
        with col2:
            st.metric("AQI Range", f"{selected_aqi}")
        with col3:
            st.metric("Health Status", health_info.get('health_status', 'Unknown'))
        
        # Health Effects
        st.markdown("#### 🏥 Health Effects")
        for effect in health_info.get('health_effects', []):
            st.markdown(effect)
        
        # Prevention Measures by Category
        st.markdown("#### 🛡️ Prevention Measures")
        
        prevention_measures = guide.get_all_prevention_measures(aqi_category)
        
        col1, col2 = st.columns(2)
        categories_list = list(prevention_measures.keys())
        
        with col1:
            for cat in categories_list[:2]:
                st.markdown(f"**{cat.title()}**")
                for measure in prevention_measures[cat][:3]:
                    st.markdown(measure)
        
        with col2:
            for cat in categories_list[2:]:
                st.markdown(f"**{cat.title()}**")
                for measure in prevention_measures[cat][:3]:
                    st.markdown(measure)
        
        # Population-specific guidance
        st.markdown("#### 👥 Guidance by Population Group")
        
        pop_groups = health_info.get('population_groups', {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**General Population**")
            st.info(pop_groups.get('general', ''))
            st.markdown("**Children**")
            st.warning(pop_groups.get('children', ''))
        
        with col2:
            st.markdown("**Elderly**")
            st.warning(pop_groups.get('elderly', ''))
            st.markdown("**People with Respiratory Disease**")
            st.error(pop_groups.get('respiratory', ''))
    
    # ===== TAB 4: DATA DASHBOARD =====
    with tabs[3]:
        st.markdown("<div class='section-header'>📊 AQI Data Dashboard</div>", unsafe_allow_html=True)
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        all_stats_list = list(predictor.location_stats.values())
        all_aqi_values = [s['avg_aqi'] for s in all_stats_list]
        
        with col1:
            st.metric("Total Stations", len(predictor.location_stats))
        with col2:
            st.metric("Average AQI", f"{np.mean(all_aqi_values):.1f}")
        with col3:
            st.metric("Median AQI", f"{np.median(all_aqi_values):.1f}")
        with col4:
            st.metric("Max AQI", f"{np.max(all_aqi_values):.1f}")
        
        # Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### AQI Distribution Across All Stations")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_aqi_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(all_aqi_values), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(100, color='orange', linestyle='--', linewidth=2, label='Safety Threshold')
            ax.set_xlabel('AQI Index')
            ax.set_ylabel('Number of Stations')
            ax.set_title('AQI Distribution')
            ax.legend()
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### AQI Category Distribution")
            
            categories = {'Good': 0, 'Satisfactory': 0, 'Moderate': 0, 'Poor': 0, 'Very Poor': 0, 'Severe': 0}
            for aqi in all_aqi_values:
                cat = guide.get_aqi_category(aqi)
                if cat in categories:
                    categories[cat] += 1
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_pie = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1', '#721c24']
            ax.pie([v for v in categories.values() if v > 0], 
                   labels=[k for k, v in categories.items() if v > 0],
                   autopct='%1.1f%%',
                   colors=colors_pie[:len([v for v in categories.values() if v > 0])])
            ax.set_title('Station Distribution by AQI Category')
            st.pyplot(fig, use_container_width=True)
    
    # ===== TAB 5: HEALTH IMPACTS =====
    with tabs[4]:
        st.markdown("<div class='section-header'>🏥 Health Impacts and Clinical Recommendations</div>", unsafe_allow_html=True)
        
        # Disease-specific impacts
        diseases = {
            'Asthma': {
                'emoji': '🫁',
                'impacts': {
                    'Good': 'No symptoms. All outdoor activities safe.',
                    'Satisfactory': 'Minimal risk. Occasional symptoms possible in sensitive individuals.',
                    'Moderately Polluted': 'Increased symptoms. Use inhalers as needed.',
                    'Poor': 'Frequent symptoms. Stay indoors. Medical consultation recommended.',
                    'Very Poor': 'Severe exacerbation likely. Emergency medical help may be needed.',
                    'Severe': 'LIFE-THREATENING. Seek immediate medical attention.'
                }
            },
            'Cardiovascular Disease': {
                'emoji': '❤️',
                'impacts': {
                    'Good': 'Safe for normal activities including exercise.',
                    'Satisfactory': 'Safe with normal precautions.',
                    'Moderately Polluted': 'Avoid strenuous outdoor activities.',
                    'Poor': 'Minimize exertion. Stay indoors with proper monitoring.',
                    'Very Poor': 'High risk of events. Medical standby essential.',
                    'Severe': 'CRITICAL. Emergency services on standby. Seek hospitalization.'
                }
            },
            'Chronic Respiratory Disease': {
                'emoji': '🫁',
                'impacts': {
                    'Good': 'No restrictions. Full activity safe.',
                    'Satisfactory': 'Monitor symptoms. Limit prolonged exposure.',
                    'Moderately Polluted': 'Use prescribed medications. Limit outdoor time.',
                    'Poor': 'Use N95 masks. Medical help on standby.',
                    'Very Poor': 'Use N100 masks. Hospital-grade air filters essential.',
                    'Severe': 'CRITICAL CONDITION. Professional medical care required.'
                }
            },
            'Diabetes': {
                'emoji': '🩺',
                'impacts': {
                    'Good': 'No special precautions.',
                    'Satisfactory': 'Standard activity. Monitor blood sugar levels.',
                    'Moderately Polluted': 'Avoid vigorous outdoor exercise.',
                    'Poor': 'Minimize outdoor exposure. Regular glucose monitoring.',
                    'Very Poor': 'Stay indoors. Increase glucose monitoring frequency.',
                    'Severe': 'Emergency medical protocol. Hospital-ready.'
                }
            }
        }
        
        # Select disease
        selected_disease = st.selectbox("🏥 Select Disease/Condition", list(diseases.keys()))
        disease_info = diseases[selected_disease]
        
        st.markdown(f"#### {disease_info['emoji']} {selected_disease} - AQI Impact Chart")
        
        # Health impacts by AQI level
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**AQI Level**")
            for level in ['Good', 'Satisfactory', 'Moderately Polluted', 'Poor', 'Very Poor', 'Severe']:
                st.markdown(f"`{level}`")
        
        with col2:
            st.markdown("**Health Impact & Recommendations**")
            for level, impact in disease_info['impacts'].items():
                color = {
                    'Good': 'success',
                    'Satisfactory': 'info',
                    'Moderately Polluted': 'warning',
                    'Poor': 'danger',
                    'Very Poor': 'danger',
                    'Severe': 'danger'
                }.get(level, 'secondary')
                
                st.markdown(f":{color}[{level}]: {impact}")
        
        # Risk management
        st.markdown("#### 🛡️ Risk Management Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Preventive Measures**")
            st.markdown("""
            - Maintain regular exercise during good air quality days
            - Take prescribed medications as directed
            - Keep emergency medications accessible
            - Monitor symptoms daily
            - Maintain healthy diet and hydration
            - Get adequate sleep
            - Manage stress levels
            """)
        
        with col2:
            st.markdown("**Monitoring Protocol**")
            st.markdown("""
            - Check AQI forecast daily
            - Monitor local air quality alerts
            - Track your symptoms in a log
            - Maintain regular doctor appointments
            - Keep emergency contact numbers ready
            - Ensure air filtration systems working
            - Have backup power for medical devices
            """)
        
        # Emergency contacts and resources
        st.markdown("#### 🆘 Emergency Resources")
        st.info("""
        **When to Seek Emergency Help:**
        - Severe difficulty breathing
        - Chest pain or pressure
        - Loss of consciousness
        - Uncontrolled coughing
        - Bluish lips or face
        - Severe confusion or dizziness
        
        **Contact Emergency Services:** Call 112 (India) or your local emergency number
        """)


if __name__ == "__main__":
    main()
