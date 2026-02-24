# Random Forest AQI Prediction System - Quick Reference Guide

## 📊 Project Overview
This AI project uses **Random Forest Machine Learning** to predict Air Quality Index (AQI) and provide:
1. **AQI Risk Categories** - Classify air quality from Safe to Hazardous
2. **Health Effects** - Display harmful impacts for each AQI level
3. **Improvement Measures** - Suggest specific actions to reduce pollution
4. **Early Warnings** - Alert systems for hazardous air quality

---

## 🎯 AQI Categories & Health Status

### 🟢 GOOD (0-50) - SAFE
- **Status:** Excellent air quality
- **Health Effects:** No adverse health effects for any population group
- **Action:** Enjoy outdoor activities freely

### 🟡 SATISFACTORY (51-100) - SAFE
- **Status:** Acceptable air quality
- **Health Effects:** Minor respiratory symptoms in sensitive groups only
- **Action:** Sensitive individuals should limit prolonged outdoor exposure

### 🟠 MODERATELY POLLUTED (101-150) - MODERATE HAZARD
- **Status:** Unhealthy for sensitive groups
- **Health Effects:** Respiratory discomfort, asthma attacks, coughing
- **Action:** Children, elderly, and those with respiratory disease should avoid outdoors

### 🔴 POOR (151-200) - HAZARDOUS
- **Status:** Unhealthy for general population
- **Health Effects:** Severe respiratory illness, increased heart disease, hospital admissions
- **Action:** Most people should stay indoors; Use N95 masks if going outside

### 🟣 VERY POOR (201-300) - HAZARDOUS
- **Status:** Very unhealthy, health emergency
- **Health Effects:** Life-threatening respiratory illness, severe cardiovascular issues, mass illness
- **Action:** All should stay indoors with air filters; Emergency preparedness essential

### 💀 SEVERE (301+) - HAZARDOUS
- **Status:** Hazardous, emergency conditions
- **Health Effects:** Life-threatening for all, respiratory failure, mortality risk
- **Action:** Total lockdown recommended; Stay indoors with advanced air filtration

---

## 🔬 Key Pollutants & Their Sources

| Pollutant | Source | Health Impact | Solution |
|-----------|--------|---------------|----------|
| **PM2.5** | Industries, vehicles, construction | Lung damage, asthma, cardiac issues | Air purifiers, emission controls, EV transition |
| **PM10** | Dust, roads, construction | Respiratory discomfort, reduced lung function | Street sweeping, water spraying, dust barriers |
| **NO2** | Vehicle emissions, Industries | Respiratory damage, reduced immunity | Promote public transport, fuel quality improvement |
| **CO** | Vehicle exhaust | Reduces oxygen in blood, heart stress | Improve fuel efficiency, vehicle maintenance |
| **SO2** | Thermal power plants, industries | Respiratory problems, acid rain | Use low-sulfur fuel, install scrubbers |
| **O3** | Secondary pollutant from NOx + VOC | Respiratory damage, asthma | Reduce NOx and VOC emissions |

---

## 🎯 Top Recommendations to Reduce AQI

### PRIORITY 1: Reduce PM2.5 (CRITICAL)
1. ✅ Install HEPA air purifiers in homes and offices
2. ✅ Enforce industrial emission standards strictly
3. ✅ Promote electric vehicles and reduce diesel use
4. ✅ Control construction dust with barriers and water spraying
5. ✅ Ban agricultural stubble burning
6. ✅ Implement green building standards

### PRIORITY 2: Control PM10 (CRITICAL)
1. ✅ Regular street sweeping and wet cleaning
2. ✅ Dust suppression at construction sites
3. ✅ Restrict unpaved road traffic
4. ✅ Establish vegetative cover in open areas
5. ✅ Cover mining stockpiles and storage areas

### PRIORITY 3: Reduce Vehicle Emissions (HIGH)
1. ✅ Promote public transportation expansion
2. ✅ Encourage carpooling and ride-sharing apps
3. ✅ Implement odd-even vehicle schemes
4. ✅ Support work-from-home policies
5. ✅ Provide incentives for electric vehicles
6. ✅ Improve road infrastructure for smooth traffic flow

### PRIORITY 4: Industrial Emission Control (HIGH)
1. ✅ Enforce green manufacturing practices
2. ✅ Install pollution control equipment (scrubbers, filters)
3. ✅ Relocate highly polluting industries
4. ✅ Implement strict industry-wise pollution limits
5. ✅ Regular stack emission monitoring

### PRIORITY 5: Reduce NO2 (HIGH)
1. ✅ Control vehicle emissions through testing
2. ✅ Improve fuel quality standards
3. ✅ Mandate regular vehicle maintenance
4. ✅ Reduce diesel vehicle usage
5. ✅ Promote renewable energy adoption

---

## 📈 Model Performance

- **Classifier Accuracy:** 44.7% (predicting AQI categories)
- **Regressor R² Score:** Robust feature importance analysis
- **Features Used:** 13 key environmental and pollution indicators
- **Data Points:** 10,000+ records across 24 Indian cities
- **Time Period:** 2019-2024

---

## 🚀 How to Use the Model

### Running the Python Script:
```bash
python aqi_prediction_system.py
```

### Using the Jupyter Notebook:
1. Open `AQI_RandomForest_Prediction.ipynb`
2. Run all cells in sequence
3. Modify input parameters to test different scenarios
4. View predictions and health impact warnings

### Making Predictions:
```python
display_aqi_prediction(
    city_name="Your City",
    pm25=50,      # PM2.5 concentration
    pm10=100,     # PM10 concentration
    no2=40,       # NO2 concentration
    co=0.8,       # CO concentration
    so2=15,       # SO2 concentration
    o3=30,        # O3 concentration
    temp=28,      # Temperature in °C
    humidity=60,  # Humidity in %
    wind=5,       # Wind speed in km/h
    rainfall=0,   # Rainfall in mm
    pressure=1013,    # Pressure in hPa
    vehicles=100000,  # Number of vehicles
    industrial=5.5    # Industrial activity index
)
```

---

## 📁 Project Files

- **`aqi_prediction_system.py`** - Standalone Python script with trained models
- **`AQI_RandomForest_Prediction.ipynb`** - Interactive Jupyter notebook
- **`files/aqi_cleaned.csv`** - Cleaned AQI dataset (235,785 records)
- **`files/indian_aqi_health_impact_2019_2024_cleaned.csv`** - Health impact dataset (10,000 records)

---

## 💡 Key Insights

1. **PM2.5 and PM10** are the most critical pollutants affecting AQI
2. **Vehicle emissions** and **industrial activity** are major contributors
3. **Weather factors** (temperature, humidity, wind) significantly influence AQI
4. **Multiple interventions** needed - single measures insufficient
5. **Immediate action required** for cities already in "Poor" category and above

---

## 🔔 Public Health Alerts

The system can generate automatic alerts:
- ⚠️ Yellow Alert: Satisfactory category - advisory for sensitive groups
- 🔴 Red Alert: Poor category - general public health warning
- 🚨 Critical Alert: Very Poor/Severe - emergency measures activation

---

## 🎓 Educational Value

This project demonstrates:
- Random Forest classification and regression
- Feature importance analysis for policy recommendations  
- Data-driven health impact assessment
- Environmental data analysis and visualization
- Real-world AI application for public health

---

## 📞 Support & Documentation

For questions or modifications:
1. Check the notebook for detailed explanations
2. Review the Python script for model implementation
3. Examine the health effects database for category details
4. Analyze feature importance for understanding AQI drivers

---

**Last Updated:** February 2026
**Status:** ✅ Ready for Deployment
