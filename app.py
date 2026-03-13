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
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import warnings

warnings.filterwarnings('ignore')

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="air",
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
        'status': 'SAFE',
        'severity': 1,
        'color': '#28a745',
        'emoji': '🟢',
        'health_effects': [
            'No adverse health effects',
            'Excellent air quality for all activities',
            'Safe for all population groups'
        ],
        'precautions': 'Enjoy outdoor activities freely'
    },
    'Satisfactory': {
        'range': '51-100',
        'status': 'SAFE',
        'severity': 2,
        'color': '#ffc107',
        'emoji': '🟡',
        'health_effects': [
            'Minor respiratory symptoms in sensitive groups',
            'Minimal risk for general population',
            'Occasional asthma symptoms'
        ],
        'precautions': 'Sensitive individuals should limit prolonged outdoor exposure'
    },
    'Moderately Polluted': {
        'range': '101-150',
        'status': 'MODERATE HAZARD',
        'severity': 3,
        'color': '#fd7e14',        'emoji': '🟠',        'health_effects': [
            'Respiratory discomfort for sensitive groups',
            'Increased asthma attacks and coughing',
            'Difficulty breathing for children & elderly'
        ],
        'precautions': 'Children, elderly, & people with respiratory disease should avoid outdoor activities'
    },
    'Poor': {
        'range': '151-200',
        'status': 'HAZARDOUS',
        'severity': 4,
        'color': '#dc3545',        'emoji': '🔴',        'health_effects': [
            'Severe respiratory illness in general population',
            'Increased heart disease risk',
            'Hospital admissions likely to increase'
        ],
        'precautions': 'Most people stay indoors; Use N95 masks if outside'
    },
    'Very Poor': {
        'range': '201-300',
        'status': 'HAZARDOUS',
        'severity': 5,
        'color': '#6f42c1',
        'emoji': '🟣',
        'health_effects': [
            'Life-threatening respiratory illness',
            'Severe cardiovascular complications',
            'Hospital emergencies & mortality risk'
        ],
        'precautions': 'All should stay indoors with air filters; Emergency preparedness essential'
    },
    'Severe': {
        'range': '301+',
        'status': 'HAZARDOUS',
        'severity': 6,
        'color': '#721c24',        'emoji': '💀',        'health_effects': [
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
            'Install air purifiers and HEPA filters in homes',
            'Reduce industrial emissions enforcement',
            'Promote electric vehicles over diesel/petrol',
            'Control construction dust generation',
            'Implement stricter pollution standards'
        ]
    },
    'PM10': {
        'priority': 'CRITICAL',
        'measures': [
            'Street sweeping and wet cleaning of roads',
            'Dust control at construction sites',
            'Water spraying to reduce dust particles',
            'Vegetative cover on open ground'
        ]
    },
    'Vehicle Count': {
        'priority': 'HIGH',
        'measures': [
            'Promote public transportation (buses, metro)',
            'Encourage carpooling and ride-sharing',
            'Implement odd-even vehicle schemes',
            'Work from home policies'
        ]
    },
    'Industrial Activity Index': {
        'priority': 'HIGH',
        'measures': [
            'Enforce green manufacturing practices',
            'Install scrubbers on industrial chimneys',
            'Regular stack emission monitoring'
        ]
    },
    'NO2': {
        'priority': 'HIGH',
        'measures': [
            'Control vehicle emissions at source',
            'Improve fuel quality standards',
            'Regular vehicle maintenance campaigns'
        ]
    }
}

# ===== LOAD & CACHE DATA & MODELS =====
@st.cache_resource
def load_data_and_train_models():
    """Load data and train models (fast path used on all pages)."""
    df = pd.read_csv("https://drive.google.com/uc?export=download&id=1QWBPeGqk1zVtpv6xLvJ3ynIBbOZ6rZYE")
    
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
    # Numeric month index so model can learn seasonality (1–12)
    df['MonthIndex'] = (
        df['Month']
        .astype(str)
        .str.split('.')
        .str[0]
        .astype(int)
    )
    # Add cyclic month encoding for better seasonal patterns
    df['Month_sin'] = np.sin(2 * np.pi * df['MonthIndex'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['MonthIndex'] / 12)
    
    # Features - using available columns from dataset
    features = [
        'PM2.5',
        'PM10',
        'NO',
        'NO2',
        'NOx',
        'NH3',
        'CO',
        'SO2',
        'O3',
        'Benzene',
        'Toluene',
        'Xylene',
        'Year',
        'MonthIndex',
        'Month_sin',
        'Month_cos',
    ]
    
    # Keep original df with City info and missing values for city selection
    df_original = df.copy()
    
    # Remove rows with missing values for model training only
    df_clean = df[features + ['AQI', 'AQI_Category']].dropna()
    
    X = df_clean[features]
    y_category = df_clean['AQI_Category']
    y_aqi = df_clean['AQI']
    
    # Train classifier (optimized for speed)
    X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    
    # Train regressor (optimized for speed)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_aqi, test_size=0.2, random_state=42)
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_regressor.fit(X_train_reg, y_train_reg)
    
    # Keep train/test sizes for later metrics display
    data_info = {
        'n_train': len(X_train),
        'n_test': len(X_test),
    }
    
    return df_original, rf_classifier, rf_regressor, features, data_info


@st.cache_resource
def compute_classification_metrics(df, rf_classifier, features, data_info):
    """Compute full classification metrics (only when metrics page is opened)."""
    df_clean = df[features + ['AQI', 'AQI_Category']].dropna()
    X = df_clean[features]
    y_category = df_clean['AQI_Category']
    
    _, X_test, _, y_test = train_test_split(
        X, y_category, test_size=0.2, random_state=42
    )
    y_pred = rf_classifier.predict(X_test)
    labels = sorted(y_category.unique())
    clf_report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Also print to terminal so you can see it in the console
    print("\n=== Classification Report (AQI Category) ===")
    print(clf_report)
    print("\n=== Confusion Matrix (rows = true, cols = predicted) ===")
    print(labels)
    print(cm)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_weighted': precision_score(
            y_test, y_pred, average='weighted', zero_division=0
        ),
        'recall_weighted': recall_score(
            y_test, y_pred, average='weighted', zero_division=0
        ),
        'f1_weighted': f1_score(
            y_test, y_pred, average='weighted', zero_division=0
        ),
        'classification_report': clf_report,
        'confusion_matrix': cm,
        'labels': labels,
        'n_train': data_info.get('n_train', None),
        'n_test': data_info.get('n_test', None),
    }
    return metrics

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
    st.title("Air Quality Index (AQI) Prediction System")
    st.markdown("### Powered by Random Forest Machine Learning")
    
    # Load data and models (fast path)
    df, rf_classifier, rf_regressor, features, data_info = load_data_and_train_models()
    
    # Sidebar navigation
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Predict AQI", "Data Analysis", "Model Metrics", "Top Recommendations", "About"]
    )
    
    if page == "Home":
        st.header("Welcome to AQI Prediction System")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Cities Analyzed", df['City'].nunique())
        with col3:
            st.metric("AQI Range", f"{df['AQI'].min():.0f} - {df['AQI'].max():.0f}")
        
        st.markdown("---")
        st.markdown("""
        ### What is this System?
        
        This AI-powered system uses **Random Forest Machine Learning** to:
        -  **Predict AQI categories** (Good → Severe)
        -  **Classify air quality** as SAFE or HAZARDOUS
        -  **Show health effects** for each AQI level
        -  **Suggest improvement measures** to reduce pollution
        
        ### Three Ways to Use This App:
        
        #### 1. Quick Scenarios (No Data Entry)
        > Perfect if you don't have pollutant measurements
        - Select from 4 preset scenarios: Good, Moderate, Poor, Severe
        - Get instant predictions with one click
        - Great for understanding AQI impact levels
        
        #### 2. Real City Data (Most Practical)
        > Use actual data from our dataset
        - Select your city from dropdown
        - Auto-loads historical average pollutant values
        - No manual entry needed!
        - Works for 24 Indian cities
        
        #### 3. Custom Parameters (Advanced)
        > For detailed analysis
        - Use sliders to adjust values
        - Compare different scenarios
        - For researchers and analysts
        """)
        
        # Show category guide
        st.markdown("### AQI Categories at a Glance")
        
        categories_data = []
        for cat in ['Good', 'Satisfactory', 'Moderately Polluted', 'Poor', 'Very Poor', 'Severe']:
            info = HEALTH_EFFECTS_DB[cat]
            categories_data.append({
                'Category': cat,
                'Range': info['range'],
                'Status': info['status']
            })
        
        cat_df = pd.DataFrame(categories_data)
        st.table(cat_df)
    
    elif page == "Predict AQI":
        st.header("Predict AQI for Your City")
        
        # Quick scenarios
        st.markdown("### Quick Scenarios")
        
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
            st.info(f"Using preset: **{selected_scenario}**")
        else:
            st.markdown("### Enter or Adjust Values")
        
        # Input columns with sliders for better UX (only show if not using loaded city data)
        if not st.session_state.loaded_city_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if use_preset:
                    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, preset_values['pm25'])
                    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, preset_values['pm10'])
                    no = st.slider("NO (ppb)", 0.0, 500.0, 10.0)
                else:
                    pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, 50.0)
                    pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 100.0)
                    no = st.slider("NO (ppb)", 0.0, 500.0, 10.0)
            
            with col2:
                if use_preset:
                    no2 = st.slider("NO2 (ppb)", 0.0, 500.0, preset_values['no2'])
                    nox = st.slider("NOx (ppb)", 0.0, 500.0, 30.0)
                    nh3 = st.slider("NH3 (ppb)", 0.0, 500.0, 20.0)
                else:
                    no2 = st.slider("NO2 (ppb)", 0.0, 500.0, 40.0)
                    nox = st.slider("NOx (ppb)", 0.0, 500.0, 30.0)
                    nh3 = st.slider("NH3 (ppb)", 0.0, 500.0, 20.0)
            
            with col3:
                if use_preset:
                    co = st.slider("CO (ppm)", 0.0, 5.0, preset_values['co'])
                    so2 = st.slider("SO2 (ppb)", 0.0, 200.0, preset_values['so2'])
                    o3 = st.slider("O3 (ppb)", 0.0, 500.0, preset_values['o3'])
                else:
                    co = st.slider("CO (ppm)", 0.0, 5.0, 1.0)
                    so2 = st.slider("SO2 (ppb)", 0.0, 200.0, 20.0)
                    o3 = st.slider("O3 (ppb)", 0.0, 500.0, 40.0)
            
            with col4:
                if use_preset:
                    benzene = st.slider("Benzene (µg/m³)", 0.0, 500.0, 3.0)
                    toluene = st.slider("Toluene (µg/m³)", 0.0, 500.0, 10.0)
                    xylene = st.slider("Xylene (µg/m³)", 0.0, 500.0, 2.0)
                    year = st.slider("Year", 2015, 2024, 2020)
                    month_index_input = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, 6)
                else:
                    benzene = st.slider("Benzene (µg/m³)", 0.0, 500.0, 3.0)
                    toluene = st.slider("Toluene (µg/m³)", 0.0, 500.0, 10.0)
                    xylene = st.slider("Xylene (µg/m³)", 0.0, 500.0, 2.0)
                    year = st.slider("Year", 2015, 2024, 2020)
                    month_index_input = st.slider("Month (1 = Jan, 12 = Dec)", 1, 12, 6)
        
        st.markdown("---")
        st.markdown("### OR Use Real City Data from Dataset")
        
        # Get unique cities from dataset
        cities_in_data = sorted(df['City'].unique().tolist())
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_city_data = st.selectbox("Select a city:", cities_in_data)
        
        with col2:
            if st.button("Load Data", use_container_width=True):
                # Get MEDIAN/AVERAGE values for the selected city (not just first row)
                city_data_all = df[df['City'] == selected_city_data]
                city_avg = city_data_all.median(numeric_only=True)  # Use median to avoid outliers
                
                # Get global median as fallback for missing values
                global_avg = df.median(numeric_only=True)
                
                # Fill NaN values with global median
                for col in ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']:
                    if pd.isna(city_avg[col]):
                        city_avg[col] = global_avg.get(col, 50)
                
                # Get the actual median AQI for display
                median_aqi = city_data_all['AQI'].median()
                if pd.isna(median_aqi):
                    # If city has no AQI data, use global median
                    median_aqi = df['AQI'].median()
                
                st.session_state.loaded_city_data = {
                    'pm25': float(city_avg['PM2.5']),
                    'pm10': float(city_avg['PM10']),
                    'no': float(city_avg['NO']),
                    'no2': float(city_avg['NO2']),
                    'nox': float(city_avg['NOx']),
                    'nh3': float(city_avg['NH3']),
                    'co': float(city_avg['CO']),
                    'so2': float(city_avg['SO2']),
                    'o3': float(city_avg['O3']),
                    'benzene': float(city_avg['Benzene']),
                    'toluene': float(city_avg['Toluene']),
                    'xylene': float(city_avg['Xylene']),
                    'year': int(city_avg.get('Year', 2020)),
                    'city_name': selected_city_data,
                    'median_aqi': int(median_aqi)
                }
                st.success(f"Loaded real data for {selected_city_data} (Median AQI: {int(median_aqi)})")
                st.rerun()
        
        st.markdown("---")
        
        # Use loaded city data if available, otherwise use preset values
        if st.session_state.loaded_city_data:
            city_data = st.session_state.loaded_city_data
            pm25 = city_data['pm25']
            pm10 = city_data['pm10']
            no = city_data['no']
            no2 = city_data['no2']
            nox = city_data['nox']
            nh3 = city_data['nh3']
            co = city_data['co']
            so2 = city_data['so2']
            o3 = city_data['o3']
            benzene = city_data['benzene']
            toluene = city_data['toluene']
            xylene = city_data['xylene']
            year = city_data['year']
            city_name = city_data['city_name']
            st.info(f"Using data for {city_name} (loaded from dataset)")
        
        # Month-wise AQI prediction using dataset
        st.markdown("---")
        st.markdown("### Predict AQI by Month (from Dataset)")
        
        colm1, colm2, colm3 = st.columns([2, 2, 1])
        with colm1:
            month_city = st.selectbox(
                "Select city for month-wise prediction:",
                sorted(df['City'].unique()),
                key="month_city",
            )
        with colm2:
            month_label = st.selectbox(
                "Select month:",
                sorted(df['Month'].unique()),
                key="month_label",
            )
        with colm3:
            use_actual_median = st.checkbox(
                "Use actual median AQI (from data)", value=False, key="use_actual_median"
            )
        
        if st.button("Predict Month-wise AQI", use_container_width=True):
            month_subset = df[(df['City'] == month_city) & (df['Month'] == month_label)]
            if month_subset.empty:
                st.warning("No data available for the selected city and month.")
            else:
                month_median = month_subset.median(numeric_only=True)
                # MonthIndex for the selected month (1–12)
                month_index_val = int(str(month_label).split('.')[0])

                feature_values_month = {
                    'PM2.5': float(month_median['PM2.5']),
                    'PM10': float(month_median['PM10']),
                    'NO': float(month_median['NO']),
                    'NO2': float(month_median['NO2']),
                    'NOx': float(month_median['NOx']),
                    'NH3': float(month_median['NH3']),
                    'CO': float(month_median['CO']),
                    'SO2': float(month_median['SO2']),
                    'O3': float(month_median['O3']),
                    'Benzene': float(month_median['Benzene']),
                    'Toluene': float(month_median['Toluene']),
                    'Xylene': float(month_median['Xylene']),
                    'Year': int(month_median.get('Year', 2020)),
                    'MonthIndex': month_index_val,
                }
                
                # Add cyclic month encoding for seasonal pattern capture
                feature_values_month['Month_sin'] = np.sin(2 * np.pi * month_index_val / 12)
                feature_values_month['Month_cos'] = np.cos(2 * np.pi * month_index_val / 12)
                
                if use_actual_median:
                    aqi_value = float(month_subset['AQI'].median())
                    aqi_category = categorize_aqi(aqi_value)
                else:
                    input_values_month = [feature_values_month[f] for f in features]
                    aqi_value, aqi_category = predict_aqi(
                        rf_regressor, rf_classifier, features, input_values_month
                    )
                
                st.markdown("---")
                st.markdown(f"### Month-wise Prediction for {month_city} ({month_label})")
                
                colm_res1, colm_res2, colm_res3 = st.columns(3)
                effect_info_month = HEALTH_EFFECTS_DB[aqi_category]
                
                with colm_res1:
                    st.metric("AQI Value", f"{aqi_value:.1f}")
                with colm_res2:
                    st.metric("Category", aqi_category)
                with colm_res3:
                    st.metric("Status", effect_info_month['status'])
                
                st.markdown(f"**Health Effects ({aqi_category})**")
                for effect in effect_info_month['health_effects']:
                    st.write(effect)
                st.warning(f"**Precautions:** {effect_info_month['precautions']}")
        
        # Make prediction
        if st.button("Predict AQI", use_container_width=True):
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
                # Create input array in the correct order of features
                feature_values = {
                    'PM2.5': pm25,
                    'PM10': pm10,
                    'NO': no,
                    'NO2': no2,
                    'NOx': nox,
                    'NH3': nh3,
                    'CO': co,
                    'SO2': so2,
                    'O3': o3,
                    'Benzene': benzene,
                    'Toluene': toluene,
                    'Xylene': xylene,
                    'Year': year,
                    'MonthIndex': month_index_input,
                }
                
                # Add cyclic month encoding for seasonal pattern capture
                feature_values['Month_sin'] = np.sin(2 * np.pi * month_index_input / 12)
                feature_values['Month_cos'] = np.cos(2 * np.pi * month_index_input / 12)
                
                # Create input in correct order
                input_values = [feature_values[f] for f in features]
                aqi_value, aqi_category = predict_aqi(rf_regressor, rf_classifier, features, input_values)
            
            # Display prediction
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            effect_info = HEALTH_EFFECTS_DB[aqi_category]
            
            with col1:
                st.metric("AQI Value", f"{aqi_value:.1f}", delta="Air Quality")
            with col2:
                st.metric("Category", aqi_category)
            with col3:
                st.metric("Status", effect_info['status'])
            
            # Health effects
            st.markdown("### Health Impact Warnings")
            
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
            st.markdown(f"### Recommended Improvements for {city_name}")
            
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
    
    elif page == "Data Analysis":
        st.header("Data Analysis & Visualization")
        
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
    
    elif page == "Model Metrics":
        st.header("Model Evaluation Metrics")
        
        # Compute metrics lazily so they are only calculated
        # when this page is opened (avoids slowing app startup).
        model_metrics = compute_classification_metrics(df, rf_classifier, features, data_info)
        
        st.markdown("### Overall Performance on Test Data")
        colm1, colm2, colm3, colm4 = st.columns(4)
        with colm1:
            st.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
        with colm2:
            st.metric("Precision (weighted)", f"{model_metrics['precision_weighted']:.3f}")
        with colm3:
            st.metric("Recall (weighted)", f"{model_metrics['recall_weighted']:.3f}")
        with colm4:
            st.metric("F1-score (weighted)", f"{model_metrics['f1_weighted']:.3f}")
        
        st.markdown(
            f"**Train samples:** {model_metrics['n_train']} &nbsp;&nbsp; "
            f"**Test samples:** {model_metrics['n_test']}"
        )
        
        st.markdown("---")
        st.markdown("### Detailed Classification Report")
        st.text(model_metrics['classification_report'])
        
        st.markdown("---")
        st.markdown("### Confusion Matrix")
        cm = model_metrics['confusion_matrix']
        labels = model_metrics['labels']
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax_cm,
        )
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        st.pyplot(fig_cm)
    
    elif page == "Top Recommendations":
        st.header("Top Recommendations to Reduce AQI")
        
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
    
    elif page == "About":
        st.header("About this System")
        
        st.markdown("""
        ### Project Overview
        
        This **Random Forest Machine Learning** system predicts Air Quality Index (AQI) and provides:
        
        - **AQI Category Prediction** - Classifies air quality from Safe to Hazardous
        - **Health Impact Assessment** - Shows specific health effects for each category
        - **Improvement Measures** - Suggests actionable solutions based on feature importance
        - **Data-Driven Insights** - Analyzes 10,000+ records from 24 Indian cities (2019-2024)
        
        ### Key Features
        
        - Real-time AQI prediction for any city
        - Comprehensive health effect warnings
        - Evidence-based improvement recommendations
        - Interactive data visualizations
        - Historical trend analysis
        
        ### Technical Details
        
        - **Model Type:** Random Forest Classifier & Regressor
        - **Features Used:** 13 environmental and pollution indicators
        - **Data Points:** 10,000+ records
        - **Cities Covered:** 24 Indian cities
        - **Time Period:** 2019-2024
        
        ### AQI Scale
        
        | Category | Range | Status |
        |----------|-------|--------|
        | Good | 0-50 | SAFE |
        | Satisfactory | 51-100 | SAFE |
        | Moderately Polluted | 101-150 | MODERATE HAZARD |
        | Poor | 151-200 | HAZARDOUS |
        | Very Poor | 201-300 | HAZARDOUS |
        | Severe | 301+ | HAZARDOUS |
        
        ### Use Cases
        
        - **Public Health:** Early warning systems for hazardous air quality
        - **Policy Making:** Data-driven recommendations for pollution control
        - **Urban Planning:** Identify key pollution sources
        - **Research:** Study air quality patterns and trends
        - **Education:** Learn about air pollution and health impacts
        
        ### Support
        
        For more information, check the project documentation or consult the data analysis section.
        """)

if __name__ == "__main__":
    main()
