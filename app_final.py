import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AQI Spatial-Temporal Prediction", layout="wide")

st.markdown("<h1 style='color: #1f77b4;'>🌍 AQI Spatial-Temporal Prediction System</h1>", unsafe_allow_html=True)

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    """Load cleaned AQI dataset"""
    df = pd.read_csv('files/station_hour_cleaned.csv')
    return df

@st.cache_data
def get_month_number(month_str):
    """Extract month number from format like '01. Jan'"""
    try:
        return int(month_str.split('.')[0])
    except:
        return None

df = load_data()

# Add numeric month column for easier processing
df['MonthNum'] = df['Month'].apply(get_month_number)

# Extract unique cities and create city-station mapping
cities = sorted(df['City'].unique())
city_stations = {city: sorted(df[df['City'] == city]['StationName'].unique().tolist()) 
                 for city in cities}

# Month names mapping
month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
month_short = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# ===== AQI CATEGORIZATION =====
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "🟢", "#d4edda"
    elif aqi <= 100:
        return "Satisfactory", "🟡", "#fff3cd"
    elif aqi <= 150:
        return "Moderate", "🟠", "#ffe0b2"
    elif aqi <= 200:
        return "Poor", "🔴", "#ffcccc"
    elif aqi <= 300:
        return "Very Poor", "🟣", "#e1bee7"
    else:
        return "Severe", "💀", "#721c24"

# ===== TAB 1: PREDICTIONS =====
tabs = st.tabs(["🔮 Predict AQI", "🗺️ Spatial Analysis", "⚠️ Prevention Guide", "📊 Dashboard"])

with tabs[0]:
    st.markdown("<h2 style='color: #2c92a5;'>Predict Future AQI</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_city = st.selectbox("🏙️ Select City", cities, key="city_pred")
    
    with col2:
        stations = city_stations.get(selected_city, [])
        selected_station = st.selectbox("📍 Select Station", stations, key="station_pred")
    
    with col3:
        pred_year = st.selectbox("📆 Year", [2024, 2025, 2026], 1)
    
    with col4:
        pred_month = st.selectbox("📅 Month", list(range(1, 13)), 5)
    
    if st.button("🔍 Predict AQI", use_container_width=True, key="predict_btn"):
        # Get historical data for this station
        station_data = df[df['StationName'] == selected_station]
        
        if not station_data.empty:
            # Calculate overall statistics
            avg_aqi = station_data['AQI'].mean()
            std_aqi = station_data['AQI'].std()
            min_aqi = station_data['AQI'].min()
            max_aqi = station_data['AQI'].max()
            
            # Calculate SEASONAL PATTERN - Average AQI for each month from historical data
            monthly_pattern = station_data.groupby('MonthNum')['AQI'].agg(['mean', 'std', 'count']).reset_index()
            monthly_pattern = monthly_pattern.sort_values('MonthNum')
            
            # Prediction based on month average from historical data
            month_data = monthly_pattern[monthly_pattern['MonthNum'] == pred_month]
            if not month_data.empty:
                predicted_aqi = month_data['mean'].values[0]
                month_std = month_data['std'].values[0]
                month_count = month_data['count'].values[0]
            else:
                predicted_aqi = avg_aqi
                month_std = std_aqi
                month_count = 0
            
            predicted_aqi = max(0, predicted_aqi)  # No negative AQI
            
            # Display prediction metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted AQI", f"{predicted_aqi:.0f}")
            
            with col2:
                category, emoji, _ = get_aqi_category(predicted_aqi)
                st.metric("Category", f"{emoji} {category}")
            
            with col3:
                st.metric("Historical Avg", f"{avg_aqi:.0f}")
            
            with col4:
                st.metric("Seasonal Std Dev", f"{month_std:.0f}")
            
            # AQI Gauge
            st.markdown(f"#### {emoji} {category.upper()} AQI Level")
            
            fig, ax = plt.subplots(figsize=(12, 2))
            ranges = [0, 50, 100, 150, 200, 300, 350]
            colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1', '#721c24', '#000000']
            
            for i in range(len(ranges)-1):
                ax.barh(0, ranges[i+1] - ranges[i], left=ranges[i], height=0.3, color=colors[i])
            
            ax.plot([predicted_aqi, predicted_aqi], [-0.15, 0.15], 'k-', linewidth=4, marker='v')
            ax.set_xlim(0, 350)
            ax.set_ylim(-1, 1)
            ax.set_xlabel('AQI Index', fontweight='bold')
            ax.set_xticks(ranges)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig, use_container_width=True)
            
            # MONTHLY SEASONAL TREND - Show how pollution varies by month
            st.markdown("#### 📅 Seasonal AQI Trend (Based on Historical Data Pattern)")
            st.info(f"📊 Analyzing {int(month_count)} historical records for {month_names[pred_month]}")
            
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Create color array based on AQI values
            colors_trend = ['#28a745' if x <= 50 else '#ffc107' if x <= 100 else '#fd7e14' 
                           if x <= 150 else '#dc3545' if x <= 200 else '#6f42c1' if x <= 300 
                           else '#721c24' for x in monthly_pattern['mean']]
            
            # Plot bars
            bars = ax.bar(range(len(monthly_pattern)), monthly_pattern['mean'], 
                          color=colors_trend, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Highlight the selected month
            selected_idx = pred_month - 1
            if selected_idx < len(bars):
                bars[selected_idx].set_linewidth(3)
                bars[selected_idx].set_edgecolor('darkred')
            
            ax.set_xticks(range(len(monthly_pattern)))
            ax.set_xticklabels([month_short[int(m)] for m in monthly_pattern['MonthNum']], fontweight='bold')
            ax.set_ylabel('Average AQI', fontweight='bold', fontsize=12)
            ax.set_xlabel('Month', fontweight='bold', fontsize=12)
            ax.set_title(f'Monthly AQI Pattern for {selected_station} (Historical Data)', fontweight='bold', fontsize=13)
            ax.axhline(y=100, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Safe Threshold (100)')
            ax.axhline(y=150, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Moderate Threshold (150)')
            ax.grid(axis='y', alpha=0.3)
            ax.legend(loc='upper right')
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            
            # Show seasonal comparison
            st.markdown("#### 🌡️ Seasonal Insights from Dataset")
            best_month_idx = monthly_pattern['mean'].idxmin()
            worst_month_idx = monthly_pattern['mean'].idxmax()
            
            best_month_num = int(monthly_pattern.loc[best_month_idx, 'MonthNum'])
            worst_month_num = int(monthly_pattern.loc[worst_month_idx, 'MonthNum'])
            best_aqi = monthly_pattern.loc[best_month_idx, 'mean']
            worst_aqi = monthly_pattern.loc[worst_month_idx, 'mean']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"""
                ✅ **CLEANEST MONTH (Low Pollution)**
                {month_names[best_month_num]}
                Average AQI: {best_aqi:.0f}
                """)
            
            with col2:
                st.error(f"""
                ❌ **MOST POLLUTED MONTH (Peak Pollution)**
                {month_names[worst_month_num]}
                Average AQI: {worst_aqi:.0f}
                """)
            
            with col3:
                diff = worst_aqi - best_aqi
                pct_change = (diff/best_aqi*100) if best_aqi > 0 else 0
                st.warning(f"""
                📊 **SEASONAL RANGE**
                Peak - Low: {diff:.0f} AQI
                Variation: {pct_change:.0f}%
                """)
            
            # Trend interpretation
            st.markdown("#### 📈 Pattern Analysis")
            
            # Find trend
            first_half_avg = monthly_pattern[monthly_pattern['MonthNum'] <= 6]['mean'].mean()
            second_half_avg = monthly_pattern[monthly_pattern['MonthNum'] > 6]['mean'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if first_half_avg > second_half_avg:
                    trend_text = "**Summer (Jan-Jun) Higher → Winter (Jul-Dec) Lower**\n\nAir quality IMPROVES toward end of year"
                    st.info(trend_text)
                else:
                    trend_text = "**Winter (Jul-Dec) Higher → Summer (Jan-Jun) Lower**\n\nAir quality WORSENS toward end of year (Winter Pollution)"
                    st.warning(trend_text)
            
            with col2:
                current_month_aqi = predicted_aqi
                if current_month_aqi > worst_aqi * 0.9:
                    st.error(f"🚨 {month_names[pred_month]} has SEVERE pollution - Peak season!")
                elif current_month_aqi > 150:
                    st.warning(f"⚠️ {month_names[pred_month]} has HIGH pollution")
                elif current_month_aqi > 100:
                    st.info(f"📌 {month_names[pred_month]} has MODERATE pollution")
                else:
                    st.success(f"✅ {month_names[pred_month]} has GOOD air quality")
            
            # AQI Gauge
            st.markdown(f"#### {emoji} {category.upper()} AQI Level")
            
            fig, ax = plt.subplots(figsize=(12, 2))
            ranges = [0, 50, 100, 150, 200, 300, 350]
            colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1', '#721c24', '#000000']
            
            for i in range(len(ranges)-1):
                ax.barh(0, ranges[i+1] - ranges[i], left=ranges[i], height=0.3, color=colors[i])
            
            ax.plot([predicted_aqi, predicted_aqi], [-0.15, 0.15], 'k-', linewidth=4, marker='v')
            ax.set_xlim(0, 350)
            ax.set_ylim(-1, 1)
            ax.set_xlabel('AQI Index', fontweight='bold')
            ax.set_xticks(ranges)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig, use_container_width=True)
            
            # MONTHLY SEASONAL TREND - Show how pollution varies by month
            st.markdown("#### 📅 Seasonal AQI Trend (Month-wise Pattern)")
            
            fig, ax = plt.subplots(figsize=(12, 4))
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            monthly_pattern_sorted = monthly_pattern.sort_values('Month')
            colors_trend = ['#28a745' if x <= 50 else '#ffc107' if x <= 100 else '#fd7e14' 
                           if x <= 150 else '#dc3545' if x <= 200 else '#6f42c1' if x <= 300 
                           else '#721c24' for x in monthly_pattern_sorted['mean']]
            
            # Plot bars
            bars = ax.bar(range(len(monthly_pattern_sorted)), monthly_pattern_sorted['mean'], 
                          color=colors_trend, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Highlight the selected month
            selected_idx = pred_month - 1
            if selected_idx < len(bars):
                bars[selected_idx].set_linewidth(3)
                bars[selected_idx].set_edgecolor('black')
            
            ax.set_xticks(range(len(monthly_pattern_sorted)))
            ax.set_xticklabels([month_names[int(m)-1] for m in monthly_pattern_sorted['Month']], fontweight='bold')
            ax.set_ylabel('Average AQI', fontweight='bold')
            ax.set_title(f'Monthly AQI Variation for {selected_station}', fontweight='bold', fontsize=12)
            ax.axhline(y=100, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Safe Threshold')
            ax.axhline(y=150, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Moderate Threshold')
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=True)
            
            # Show seasonal comparison
            st.markdown("#### 🌡️ Seasonal Comparison")
            best_month_idx = monthly_pattern_sorted['mean'].idxmin()
            worst_month_idx = monthly_pattern_sorted['mean'].idxmax()
            
            best_month_num = int(monthly_pattern_sorted.loc[best_month_idx, 'Month'])
            worst_month_num = int(monthly_pattern_sorted.loc[worst_month_idx, 'Month'])
            best_aqi = monthly_pattern_sorted.loc[best_month_idx, 'mean']
            worst_aqi = monthly_pattern_sorted.loc[worst_month_idx, 'mean']
            
            month_names_full = ['January', 'February', 'March', 'April', 'May', 'June', 
                               'July', 'August', 'September', 'October', 'November', 'December']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"""
                ✅ **CLEANEST MONTH**
                {month_names_full[best_month_num-1]}
                Average AQI: {best_aqi:.0f}
                """)
            
            with col2:
                st.error(f"""
                ❌ **MOST POLLUTED MONTH**
                {month_names_full[worst_month_num-1]}
                Average AQI: {worst_aqi:.0f}
                """)
            
            with col3:
                diff = worst_aqi - best_aqi
                st.warning(f"""
                📊 **SEASONAL RANGE**
                Difference: {diff:.0f} AQI points
                Variation: {(diff/best_aqi*100):.0f}%
                """)
            
            # Details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Station Information:**")
                st.info(f"""
                Station: {selected_station}
                City: {selected_city}
                Prediction: {month_names[pred_month]} {pred_year}
                Records: {len(station_data)} historical data points
                """)
            
            with col2:
                st.markdown("**AQI Statistics:**")
                st.info(f"""
                Average: {avg_aqi:.1f}
                Min: {min_aqi:.1f}
                Max: {max_aqi:.1f}
                Std Dev: {std_aqi:.1f}
                """)
            
            # Health guidance
            st.markdown(f"#### 🏥 Health Guidance for {month_names[pred_month]}")
            
            if predicted_aqi <= 50:
                st.success("""
                ✓ Excellent air quality
                ✓ All outdoor activities safe
                ✓ No restrictions needed
                """)
            elif predicted_aqi <= 100:
                st.info("""
                ⚠ Air quality is satisfactory
                ⚠ Sensitive groups should limit prolonged outdoor exposure
                ⚠ Monitor air quality forecast
                """)
            elif predicted_aqi <= 150:
                st.warning("""
                ⛔ Respiratory symptoms possible in sensitive groups
                ⛔ Avoid strenuous outdoor activities
                ⛔ Use N95/N99 masks if going outside
                """)
            elif predicted_aqi <= 200:
                st.error("""
                🚨 Air quality is HAZARDOUS
                🚨 STAY INDOORS - minimize outdoor exposure
                🚨 Use N95/N100 masks if must go outside
                🚨 Keep medical help on standby
                """)
            else:
                st.error("""
                💀 CRITICAL - Air quality is SEVERE
                💀 LOCKDOWN MODE - No outdoor movement
                💀 All activities indoors only
                💀 Emergency protocols active
                """)
        else:
            st.error(f"No data found for {selected_station}")

# ===== TAB 2: SPATIAL ANALYSIS =====
with tabs[1]:
    st.markdown("<h2 style='color: #2c92a5;'>Spatial AQI Analysis</h2>", unsafe_allow_html=True)
    
    analysis_city = st.selectbox("🏙️ Select City for Analysis", cities, key="analysis_city")
    analysis_type = st.radio("📊 View", ["High-Risk Locations", "Safe Locations", "All Stations"])
    
    city_data = df[df['City'] == analysis_city]
    station_stats = city_data.groupby('StationName')['AQI'].agg(['mean', 'std', 'count']).reset_index()
    station_stats.columns = ['Station', 'Avg AQI', 'Std Dev', 'Count']
    station_stats = station_stats.sort_values('Avg AQI', ascending=False)
    
    if analysis_type == "High-Risk Locations":
        high_risk = station_stats[station_stats['Avg AQI'] > 150]
        
        if not high_risk.empty:
            st.warning(f"⚠️ **{len(high_risk)} High-Risk Locations Found**")
            
            for idx, row in high_risk.iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{row['Station']}**")
                with col2:
                    st.metric("Avg AQI", f"{row['Avg AQI']:.0f}")
                with col3:
                    st.metric("Records", f"{int(row['Count'])}")
            
            st.markdown("---")
            st.markdown("**❌ Recommendation**: Avoid these areas during high pollution days")
        else:
            st.success("✅ No high-risk locations in this city!")
    
    elif analysis_type == "Safe Locations":
        safe = station_stats[station_stats['Avg AQI'] < 100]
        
        if not safe.empty:
            st.success(f"✅ **{len(safe)} Safe Locations Found**")
            
            for idx, row in safe.iterrows():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{row['Station']}**")
                with col2:
                    st.metric("Avg AQI", f"{row['Avg AQI']:.0f}")
                with col3:
                    st.metric("Records", f"{int(row['Count'])}")
            
            st.markdown("---")
            st.markdown("**✅ Recommendation**: These areas have good air quality")
        else:
            st.info("No very safe locations found in this city")
    
    else:  # All stations
        st.markdown(f"**All Monitoring Stations in {analysis_city}**")
        st.dataframe(station_stats, use_container_width=True, hide_index=True)
        
        # Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_chart = ['#28a745' if x <= 50 else '#ffc107' if x <= 100 else '#fd7e14' 
                       if x <= 150 else '#dc3545' if x <= 200 else '#6f42c1' if x <= 300 
                       else '#721c24' for x in station_stats['Avg AQI']]
        
        ax.barh(station_stats['Station'], station_stats['Avg AQI'], color=colors_chart)
        ax.axvline(x=100, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Safe Threshold')
        ax.axvline(x=150, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Moderate Threshold')
        ax.set_xlabel('Average AQI', fontweight='bold')
        ax.set_title(f'AQI Distribution in {analysis_city}')
        ax.legend()
        plt.tight_layout()
        
        st.pyplot(fig, use_container_width=True)

# ===== TAB 3: PREVENTION GUIDE =====
with tabs[2]:
    st.markdown("<h2 style='color: #2c92a5;'>⚠️ Prevention & Health Guide</h2>", unsafe_allow_html=True)
    
    aqi_slider = st.slider("Select AQI Level", 0, 500, 125)
    category, emoji, color = get_aqi_category(aqi_slider)
    
    st.markdown(f"#### {emoji} {category.upper()} (AQI: {aqi_slider})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏥 Health Effects:**")
        
        if aqi_slider <= 50:
            st.success("""
            ✓ No adverse health effects
            ✓ Excellent air quality for all activity types
            ✓ Safe for children, elderly, and people with respiratory disease
            """)
        elif aqi_slider <= 100:
            st.info("""
            ⚠ Minor respiratory symptoms in sensitive groups
            ⚠ Minimal risk for general population
            ⚠ Occasional asthma symptoms
            """)
        elif aqi_slider <= 150:
            st.warning("""
            ⛔ Respiratory discomfort for sensitive groups
            ⛔ Increased asthma attacks and coughing
            ⛔ Difficulty breathing for children and elderly
            """)
        elif aqi_slider <= 200:
            st.error("""
            🚨 Severe respiratory illness in general population
            🚨 Increased heart disease risk
            🚨 Hospital admissions likely to increase
            """)
        elif aqi_slider <= 300:
            st.error("""
            🚨 Life-threatening respiratory illness
            🚨 Severe cardiovascular complications
            🚨 Hospital emergencies and mortality risk
            """)
        else:
            st.error("""
            💀 Life-threatening conditions for ALL
            💀 Respiratory failure and cardiac arrest risk
            💀 Mass casualties and mortality possible
            """)
    
    with col2:
        st.markdown("**🛡️ Prevention Steps:**")
        
        if aqi_slider <= 50:
            st.success("""
            ✓ Enjoy outdoor activities freely
            ✓ Open windows for natural ventilation
            ✓ No air purifiers needed
            """)
        elif aqi_slider <= 100:
            st.info("""
            ⚠ Sensitive individuals limit prolonged outdoor exposure
            ⚠ Consider air purifiers in bedrooms
            ⚠ Monitor daily AQI forecast
            """)
        elif aqi_slider <= 150:
            st.warning("""
            ⛔ Children/elderly avoid outdoor activities
            ⛔ Use N95 masks if going outside
            ⛔ Keep windows closed during peak hours
            ⛔ Use HEPA air filters at home
            """)
        elif aqi_slider <= 200:
            st.error("""
            🚨 Most people STAY INDOORS
            🚨 Use N95/N100 masks if absolutely necessary
            🚨 Seal windows and doors
            🚨 Run industrial-grade air purifiers
            🚨 Have emergency medical help ready
            """)
        elif aqi_slider <= 300:
            st.error("""
            🚨 CRITICAL - All groups STAY INDOORS
            🚨 Use N100 masks only
            🚨 Emergency oxygen on standby
            🚨 Evacuation protocols prepare
            """)
        else:
            st.error("""
            💀 TOTAL LOCKDOWN
            💀 No outdoor movement
            💀 Advanced filtration systems
            💀 Hospital emergency protocols active
            """)

# ===== TAB 4: DASHBOARD =====
with tabs[3]:
    st.markdown("<h2 style='color: #2c92a5;'>📊 AQI Statistics Dashboard</h2>", unsafe_allow_html=True)
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    all_aqi = df['AQI'].values
    
    with col1:
        st.metric("Total Stations", df['StationName'].nunique())
    with col2:
        st.metric("Total Cities", df['City'].nunique())
    with col3:
        st.metric("Average AQI", f"{all_aqi.mean():.1f}")
    with col4:
        st.metric("Max AQI", f"{all_aqi.max():.1f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### AQI Distribution")
        fig, ax = plt.subplots()
        ax.hist(all_aqi, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(all_aqi.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {all_aqi.mean():.0f}')
        ax.set_xlabel('AQI Index')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### AQI Category Distribution")
        
        categories = {'Good': 0, 'Satisfactory': 0, 'Moderate': 0, 'Poor': 0, 'Very Poor': 0, 'Severe': 0}
        for aqi in all_aqi:
            cat, _, _ = get_aqi_category(aqi)
            categories[cat] += 1
        
        fig, ax = plt.subplots()
        colors_pie = ['#28a745', '#ffc107', '#fd7e14', '#dc3545', '#6f42c1', '#721c24']
        ax.pie([v for v in categories.values() if v > 0],
               labels=[k for k, v in categories.items() if v > 0],
               autopct='%1.1f%%',
               colors=colors_pie[:len([v for v in categories.values() if v > 0])])
        ax.set_title('Station Distribution by AQI Category')
        st.pyplot(fig, use_container_width=True)
    
    # Top cities by AQI
    st.markdown("#### Top Cities by Average AQI")
    city_avg = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(city_avg.index, city_avg.values, color='coral')
    ax.set_xlabel('Average AQI')
    ax.set_title('Cities with Highest Average AQI')
    st.pyplot(fig, use_container_width=True)
