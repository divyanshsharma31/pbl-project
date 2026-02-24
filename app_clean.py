import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AQI Prediction System", layout="wide")
st.markdown("<h1 style='color: #1f77b4;'>🌍 AQI Spatial-Temporal Prediction System</h1>", unsafe_allow_html=True)

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    df = pd.read_csv('files/station_hour_cleaned.csv')
    df['MonthNum'] = df['Month'].apply(lambda x: int(x.split('.')[0]) if '.' in str(x) else x)
    return df

df = load_data()

# Extract unique cities
cities = sorted(df['City'].unique())
city_stations = {city: sorted(df[df['City'] == city]['StationName'].unique().tolist()) for city in cities}

# Month mapping
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
month_short = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# ===== AQI CATEGORY =====
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "🟢"
    elif aqi <= 100:
        return "Satisfactory", "🟡"
    elif aqi <= 150:
        return "Moderate", "🟠"
    elif aqi <= 200:
        return "Poor", "🔴"
    elif aqi <= 300:
        return "Very Poor", "🟣"
    else:
        return "Severe", "💀"

# ===== CREATE TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict AQI", "🗺️ Spatial Analysis", "⚠️ Prevention", "📊 Dashboard"])

# ===== TAB 1: PREDICT AQI =====
with tab1:
    st.subheader("Predict Future AQI")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sel_city = st.selectbox("City", cities)
    with col2:
        stations = city_stations[sel_city]
        sel_station = st.selectbox("Station", stations)
    with col3:
        sel_year = st.selectbox("Year", [2024, 2025, 2026], 1)
    with col4:
        sel_month = st.selectbox("Month", list(range(1, 13)), 4)
    
    if st.button("Predict", use_container_width=True):
        station_data = df[df['StationName'] == sel_station]
        
        if not station_data.empty:
            avg_aqi = station_data['AQI'].mean()
            
            # Get monthly pattern
            monthly_data = station_data[station_data['MonthNum'] == sel_month]
            if not monthly_data.empty:
                pred_aqi = monthly_data['AQI'].mean()
            else:
                pred_aqi = avg_aqi
            
            pred_aqi = max(0, pred_aqi)
            category, emoji = get_aqi_category(pred_aqi)
            
            # Display metrics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted AQI", f"{pred_aqi:.0f}")
            with col2:
                st.metric("Category", f"{emoji} {category}")
            with col3:
                st.metric("Avg AQI", f"{avg_aqi:.0f}")
            
            st.markdown("---")
            
            # Monthly trend chart
            st.subheader("Monthly Pattern (Historical)")
            monthly_pattern = station_data.groupby('MonthNum')['AQI'].mean().reset_index()
            monthly_pattern = monthly_pattern.sort_values('MonthNum')
            
            fig, ax = plt.subplots(figsize=(12, 4))
            colors = ['#28a745' if x <= 50 else '#ffc107' if x <= 100 else '#fd7e14' 
                     if x <= 150 else '#dc3545' if x <= 200 else '#6f42c1' if x <= 300 
                     else '#721c24' for x in monthly_pattern['AQI']]
            
            ax.bar(range(len(monthly_pattern)), monthly_pattern['AQI'], color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(monthly_pattern)))
            ax.set_xticklabels([month_short[int(m)] for m in monthly_pattern['MonthNum']])
            ax.set_ylabel('Average AQI')
            ax.set_title(f'Seasonal Pattern - {sel_station}')
            ax.axhline(100, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(150, color='red', linestyle='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            
            # Seasonal insights
            st.subheader("Seasonal Insights")
            best_month = monthly_pattern.loc[monthly_pattern['AQI'].idxmin()]
            worst_month = monthly_pattern.loc[monthly_pattern['AQI'].idxmax()]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"""
                **Best Month**: {month_names[int(best_month['MonthNum'])]}
                AQI: {best_month['AQI']:.0f}
                """)
            with col2:
                st.error(f"""
                **Worst Month**: {month_names[int(worst_month['MonthNum'])]}
                AQI: {worst_month['AQI']:.0f}
                """)
            with col3:
                diff = worst_month['AQI'] - best_month['AQI']
                st.info(f"""
                **Variation**: {diff:.0f} AQI
                """)

# ===== TAB 2: SPATIAL ANALYSIS =====
with tab2:
    st.subheader("Spatial AQI Analysis")
    
    anal_city = st.selectbox("Select City", cities, key="anal_city")
    anal_type = st.radio("View Type", ["High-Risk", "Safe", "All Stations"])
    
    city_data = df[df['City'] == anal_city]
    station_stats = city_data.groupby('StationName')['AQI'].agg(['mean', 'std', 'count']).reset_index()
    station_stats.columns = ['Station', 'Avg AQI', 'Std Dev', 'Count']
    station_stats = station_stats.sort_values('Avg AQI', ascending=False)
    
    st.markdown("---")
    
    if anal_type == "High-Risk":
        high_risk = station_stats[station_stats['Avg AQI'] > 150]
        if not high_risk.empty:
            st.warning(f"⚠️ {len(high_risk)} High-Risk Locations")
            for idx, row in high_risk.iterrows():
                st.metric(f"{row['Station']}", f"{row['Avg AQI']:.0f} AQI")
        else:
            st.success("No high-risk locations")
    
    elif anal_type == "Safe":
        safe = station_stats[station_stats['Avg AQI'] < 100]
        if not safe.empty:
            st.success(f"✅ {len(safe)} Safe Locations")
            for idx, row in safe.iterrows():
                st.metric(f"{row['Station']}", f"{row['Avg AQI']:.0f} AQI")
        else:
            st.info("No very safe locations")
    
    else:
        st.dataframe(station_stats, use_container_width=True, hide_index=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#28a745' if x <= 50 else '#ffc107' if x <= 100 else '#fd7e14' 
                 if x <= 150 else '#dc3545' for x in station_stats['Avg AQI']]
        ax.barh(station_stats['Station'], station_stats['Avg AQI'], color=colors, alpha=0.7)
        ax.set_xlabel('Average AQI')
        ax.set_title(f'AQI in {anal_city}')
        st.pyplot(fig, use_container_width=True)

# ===== TAB 3: PREVENTION & GOVERNMENT ACTIONS =====
with tab3:
    st.subheader("Prevention Guide & Government Policy Recommendations")
    
    aqi_level = st.slider("Select AQI Level", 0, 500, 125)
    category, emoji = get_aqi_category(aqi_level)
    
    st.markdown(f"## {emoji} {category.upper()} (AQI: {aqi_level})")
    st.markdown("---")
    
    # Health Effects
    st.subheader("🏥 Health Effects by AQI Level")
    
    health_info = {
        "Good (0-50)": {
            "general": "No adverse health effects. Air quality is excellent.",
            "children": "Safe for all outdoor activities",
            "elderly": "No restrictions needed",
            "respiratory": "No respiratory concerns"
        },
        "Satisfactory (51-100)": {
            "general": "Generally acceptable. Minimal respiratory symptoms possible.",
            "children": "Limit strenuous activities for sensitive children",
            "elderly": "Light outdoor activities safe",
            "respiratory": "Mild symptoms possible in asthmatics"
        },
        "Moderate (101-150)": {
            "general": "Unhealthy for sensitive groups. General public less affected.",
            "children": "Avoid strenuous outdoor activities",
            "elderly": "Limit outdoor exposure, use masks",
            "respiratory": "Increased asthma attacks, coughing in sensitive individuals"
        },
        "Poor (151-200)": {
            "general": "Unhealthy. General public may experience mild symptoms.",
            "children": "Avoid outdoor activities, prefer indoors",
            "elderly": "Minimize outdoor exposure completely",
            "respiratory": "Breathing difficulties, increased hospital visits"
        },
        "Very Poor (201-300)": {
            "general": "Very unhealthy for everyone. Serious health impacts likely.",
            "children": "NO outdoor activities - all indoors",
            "elderly": "Emergency medical support required",
            "respiratory": "Severe respiratory crisis, cardiovascular issues"
        },
        "Severe (301+)": {
            "general": "Hazardous. Everyone affected. Life-threatening condition.",
            "children": "Complete lockdown - all activities indoors",
            "elderly": "Life-threatening - emergency protocols",
            "respiratory": "Respiratory failure risk, mass casualties possible"
        }
    }
    
    if aqi_level <= 50:
        key = "Good (0-50)"
    elif aqi_level <= 100:
        key = "Satisfactory (51-100)"
    elif aqi_level <= 150:
        key = "Moderate (101-150)"
    elif aqi_level <= 200:
        key = "Poor (151-200)"
    elif aqi_level <= 300:
        key = "Very Poor (201-300)"
    else:
        key = "Severe (301+)"
    
    h_info = health_info[key]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"👥 **General Public**\n{h_info['general']}")
    with col2:
        st.info(f"👶 **Children**\n{h_info['children']}")
    with col3:
        st.info(f"👴 **Elderly**\n{h_info['elderly']}")
    with col4:
        st.info(f"🫁 **Respiratory Patients**\n{h_info['respiratory']}")
    
    st.markdown("---")
    
    # Public Prevention
    st.subheader("👨‍👩‍👧‍👦 Prevention Steps for PUBLIC")
    
    prevention_public = {
        "Good (0-50)": [
            "✓ Enjoy outdoor activities freely",
            "✓ Open windows for natural ventilation",
            "✓ No special precautions needed",
            "✓ Continue normal outdoor exercise"
        ],
        "Satisfactory (51-100)": [
            "⚠ Sensitive groups limit prolonged outdoor exposure",
            "⚠ Use air purifiers in bedrooms",
            "⚠ Monitor daily AQI forecast",
            "⚠ Reduce outdoor activities for children and elderly",
            "⚠ Wear N95 masks during peak hours if outdoors"
        ],
        "Moderate (101-150)": [
            "⛔ Children and elderly AVOID outdoor activities",
            "⛔ Use N95/FFP2 masks if must go outside",
            "⛔ Keep windows closed during peak hours (6-9 AM, 7-10 PM)",
            "⛔ Use HEPA air filters at home",
            "⛔ Move outdoor exercise indoors (treadmill, yoga)",
            "⛔ Avoid cooking activities that increase indoor pollution",
            "⛔ Remote work/classes for sensitive groups"
        ],
        "Poor (151-200)": [
            "🚨 LIMIT OUTDOOR ACTIVITIES",
            "🚨 Use N95/FFP2 masks if must go outside",
            "🚨 Seal windows - use AC with HEPA filters",
            "🚨 Run air purifiers in home and office",
            "🚨 Most outdoor work should be avoided",
            "🚨 Schools may offer online options",
            "🚨 All outdoor events postponed",
            "🚨 Stock masks and medications at home"
        ],
        "Very Poor (201-300)": [
            "💀 CRITICAL - Most people STAY INDOORS",
            "💀 Use N100/P100 respirators (not just N95)",
            "💀 Industrial-grade HEPA purifiers continuously",
            "💀 NO outdoor movement except emergencies",
            "💀 Schools CLOSED - online classes mandatory",
            "💀 Work from home MANDATORY",
            "💀 Keep oxygen cylinders at home",
            "💀 Hospitals: Emergency preparations"
        ],
        "Severe (301+)": [
            "☠️ TOTAL LOCKDOWN - Life-threatening air",
            "☠️ NO OUTDOOR MOVEMENT under any circumstances",
            "☠️ Advanced filtration systems ESSENTIAL (HEPA + activated carbon)",
            "☠️ Hospital emergency protocols active - ICU preparation",
            "☠️ Medical evacuation to cleaner cities URGENT",
            "☠️ All non-essential activities CANCELLED",
            "☠️ Army/Emergency services deployed",
            "☠️ Mass casualty management protocols activate"
        ]
    }
    
    for step in prevention_public[key]:
        st.write(step)
    
    st.markdown("---")
    
    # Government Actions
    st.subheader("🏛️ GOVERNMENT & POLICY ACTIONS")
    
    government_actions = {
        "Good (0-50)": [
            "✅ Maintain current pollution control standards",
            "✅ Continue regular air quality monitoring",
            "✅ Promote sustainable transportation",
            "✅ Monitor industrial emissions"
        ],
        "Satisfactory (51-100)": [
            "⚠ Issue air quality forecasts daily",
            "⚠ Advise sensitive groups (children, elderly) to limit exposure",
            "⚠ Monitor pollution sources (traffic, industries)",
            "⚠ Increase tree plantation programs",
            "⚠ Promote public transport over personal vehicles"
        ],
        "Moderate (101-150)": [
            "🚨 Issue AIR QUALITY ALERT to public",
            "🚨 Construction ban: Stop all construction activities",
            "🚨 Traffic management: Implement odd-even vehicle scheme",
            "🚨 Industries: Reduce production or stop polluting units",
            "🚨 Schools: Conduct online classes (optional closure)",
            "🚨 Public Events: Cancel outdoor gatherings",
            "🚨 Increase street cleaning and water sprinkling",
            "🚨 Deploy dust control measures"
        ],
        "Poor (151-200)": [
            "🔴 STRONG ALERT - Restrict outdoor movement",
            "🔴 CONSTRUCTION SUSPENDED - All activity stopped (except essential)",
            "🔴 TRAFFIC RESTRICTION: Implement strict odd-even or 30% reduction",
            "🔴 INDUSTRY CURBS: Close polluting units, allow 50% capacity others",
            "🔴 SCHOOLS ADVISORY: Online classes recommended",
            "🔴 OFFICES: Allow work from home",
            "🔴 Distribute masks to public at key locations",
            "🔴 HOSPITALS: Respiratory units on high alert"
        ],
        "Very Poor (201-300)": [
            "💀 CRITICAL STATE - Near complete lockdown",
            "💀 CONSTRUCTION BANNED 100% - All activity stopped",
            "💀 TRAFFIC LOCKDOWN: Vehicle ban - only essential allowed",
            "💀 INDUSTRIES 80% CLOSED - Only essential services operate",
            "💀 SCHOOLS CLOSED: All online education",
            "💀 OFFICES: Work from home MANDATORY",
            "💀 PUBLIC TRANSPORT: Free distribution",
            "💀 HOSPITALS: Activate emergency mass casualty plans"
        ],
        "Severe (301+)": [
            "☠️ NATIONAL DISASTER DECLARED",
            "☠️ COMPLETE LOCKDOWN - No movement allowed",
            "☠️ ALL ECONOMIC ACTIVITIES 100% STOPPED",
            "☠️ MARTIAL LAW may be implemented",
            "☠️ MASS EVACUATION: Emergency relocation of population",
            "☠️ NATIONAL EMERGENCY: Armed forces mobilized",
            "☠️ HOSPITALS: Disaster mode - mass casualty management",
            "☠️ INTERNATIONAL SUPPORT: Request UN humanitarian aid",
            "☠️ BORDER QUARANTINE: Prevent disease spread to other regions"
        ]
    }
    
    for action in government_actions[key]:
        st.write(action)
    
    st.markdown("---")
    
    # Long-term Solutions
    st.subheader("🌱 Long-term Solutions & Infrastructure Changes")
    
    solutions = [
        "**Transportation**:\n   • Expand metro/public transport to reduce vehicles\n   • Promote electric vehicles with tax benefits\n   • Implement strict emission norms for buses/trucks\n   • Restrict diesel vehicles in urban areas",
        
        "**Industries**:\n   • Shift polluting industries to outskirts (50+ km)\n   • Implement real-time emission monitoring systems\n   • Promote clean energy (solar, wind)\n   • Fine/closure for industries exceeding pollution limits",
        
        "**Construction**:\n   • Use green building materials\n   • Implement dust control (wet construction method)\n   • Schedule construction during low-traffic hours\n   • Ban construction in sensitive areas (hospitals, schools)",
        
        "**Agriculture**:\n   • Ban stubble burning - provide subsidies for alternatives\n   • Promote crop residue management systems\n   • Encourage mechanized collection of crop waste",
        
        "**Urban Green Spaces**:\n   • Plant 100 million trees in next 5 years\n   • Create green belts around industrial zones\n   • Expand parks and open spaces for vegetation",
        
        "**Monitoring & Tech**:\n   • Install real-time AQI monitoring stations every 5 km\n   • Use AI to predict pollution spikes\n   • Deploy pollution control robots in high-traffic areas",
        
        "**Public Awareness**:\n   • School curriculum to include air quality education\n   • Regular health camps in pollution hotspots\n   • Media campaigns on pollution dangers"
    ]
    
    for sol in solutions:
        st.info(sol)

# ===== TAB 4: DASHBOARD =====
with tab4:
    st.subheader("AQI Statistics Dashboard & City-wise Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stations", df['StationName'].nunique())
    with col2:
        st.metric("Total Cities", df['City'].nunique())
    with col3:
        st.metric("Avg AQI", f"{df['AQI'].mean():.1f}")
    with col4:
        st.metric("Max AQI", f"{df['AQI'].max():.1f}")
    
    st.markdown("---")
    
    # City-wise analysis
    st.subheader("🔴 High-Pollution Cities (Average AQI > 150)")
    city_avg = df.groupby('City')['AQI'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    city_avg = city_avg.sort_values('mean', ascending=False)
    
    high_pollution_cities = city_avg[city_avg['mean'] > 150]
    
    if not high_pollution_cities.empty:
        st.warning(f"⚠️ {len(high_pollution_cities)} cities with CRITICAL pollution levels")
        
        for idx, row in high_pollution_cities.iterrows():
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric(f"🏙️ {row['City']}", f"{row['mean']:.0f}")
            with col2:
                st.metric("Range", f"{row['min']:.0f}-{row['max']:.0f}")
            with col3:
                st.metric("Std Dev", f"{row['std']:.1f}")
            with col4:
                st.metric("Records", f"{int(row['count'])}")
            with col5:
                if row['mean'] > 200:
                    st.error("SEVERE")
                elif row['mean'] > 180:
                    st.error("POOR")
                else:
                    st.warning("MODERATE")
    
    st.markdown("---")
    
    # Analysis of worst city (Ahmedabad)
    worst_city = city_avg.iloc[0]
    st.subheader(f"📍 Detailed Analysis: {worst_city['City']} (Highest Pollution)")
    
    st.error(f"""
    ### ⚠️ CRITICAL POLLUTION ALERT
    
    **City**: {worst_city['City']}
    **Average AQI**: {worst_city['mean']:.1f} (SEVERE LEVEL)
    **Range**: {worst_city['min']:.1f} (Best) to {worst_city['max']:.1f} (Worst)
    **Standard Deviation**: {worst_city['std']:.1f}
    **Data Points**: {int(worst_city['count'])} observations
    
    **Status**: This city requires immediate government intervention and pollution control measures.
    """)
    
    # Why Ahmedabad/highest city is polluted
    st.subheader(f"🔍 Why is {worst_city['City']} So Polluted?")
    
    worst_city_data = df[df['City'] == worst_city['City']]
    worst_stations = worst_city_data.groupby('StationName')['AQI'].mean().sort_values(ascending=False).head(5)
    
    reasons = {
        "Ahmedabad": [
            "🚗 **Heavy Traffic**: Located on major highways, high vehicle density",
            "🏭 **Industrial Zones**: Multiple textile mills, chemical factories nearby",
            "🌪️ **Geography**: Semi-arid region with poor air dispersion",
            "👷 **Construction**: Rapid urban development with dust generation",
            "🌾 **Agricultural Burning**: Nearby states practice stubble burning (Oct-Nov)",
            "🏗️ **Lack of Green Space**: Limited forests and parks for air purification"
        ]
    }
    
    if worst_city['City'] in reasons:
        for reason in reasons[worst_city['City']]:
            st.info(reason)
    else:
        st.info(f"🔎 {worst_city['City']} faces pollution from multiple sources - traffic, industries, and geography")
    
    st.markdown("---")
    
    st.subheader(f"🏢 Most Polluted Stations in {worst_city['City']}")
    for station, aqi in worst_stations.items():
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"📍 **{station}**")
        with col2:
            if aqi > 200:
                st.error(f"{aqi:.0f} AQI")
            elif aqi > 150:
                st.warning(f"{aqi:.0f} AQI")
            else:
                st.info(f"{aqi:.0f} AQI")
    
    st.markdown("---")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AQI Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['AQI'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(df['AQI'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["AQI"].mean():.0f}')
        ax.axvline(150, color='orange', linestyle='--', linewidth=2, label='Moderate: 150')
        ax.axvline(200, color='darkred', linestyle='--', linewidth=2, label='Poor: 200')
        ax.set_xlabel('AQI')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 15 Cities by Average AQI")
        top_cities = city_avg.head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors_city = ['#dc3545' if x > 200 else '#fd7e14' if x > 150 else '#ffc107' if x > 100 else '#28a745' 
                      for x in top_cities['mean']]
        ax.barh(top_cities['City'], top_cities['mean'], color=colors_city, alpha=0.8, edgecolor='black')
        ax.axvline(100, color='orange', linestyle='--', alpha=0.5, label='100 (Safe)')
        ax.axvline(150, color='red', linestyle='--', alpha=0.5, label='150 (Moderate)')
        ax.set_xlabel('Average AQI')
        ax.set_title('Cities Ranked by Pollution Level')
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("📋 Government Priority Actions")
    st.error("""
    ### 🚨 IMMEDIATE ACTIONS FOR HIGH-POLLUTION CITIES
    
    **For Cities with AQI > 200 (SEVERE)**:
    1. Declare state of emergency and activate disaster management plan
    2. Implement complete vehicle ban or strict odd-even scheme  
    3. Close all non-essential industries and construction sites
    4. Distribute N95/N100 masks and medical supplies
    5. Set up temporary shelters with air purification
    6. Evacuate vulnerable populations (children, elderly, respiratory patients)
    
    **Medium-term (1-3 months)**:
    1. Phase-out coal-based industries
    2. Expand metro/public transport to reduce vehicles by 40%
    3. Relocate polluting industries outside city limits
    4. Plant 10 million trees in and around the city
    5. Implement 24/7 air quality monitoring
    
    **Long-term (1-5 years)**:
    1. Transition to 100% electric public transport
    2. Ban diesel vehicles in city centers
    3. Implement ISO 14001 certification for all industries
    4. Create green belts and parks (40% city coverage)
    5. Establish air quality as development metric for all projects
    """)

