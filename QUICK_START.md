# QUICK START GUIDE
## AQI Spatial-Temporal Prediction System

### 🚀 Get Started in 3 Steps:

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Train Models
```bash
python train_simple.py
```
(Takes 5-10 minutes)

#### 3. Run App
```bash
streamlit run app_enhanced.py
```

---

## 📍 WHAT YOU CAN DO

### Predict AQI
- Select any city and monitoring station
- Choose any month/year
- Get ensemble predictions with confidence levels
- Compare with historical data

### Find Safe/Risky Areas
- Identify high-pollution locations to avoid
- Find clean air zones to visit
- Get location-specific recommendations

### Get Health Guidance
- View health impacts at different AQI levels
- Get disease-specific recommendations
- Follow prevention steps for your health condition

### Check Prevention Steps
- By AQI level
- By population group (children, elderly, respiratory, cardiac)
- By activity type (home, outdoor, transport, work)

---

## 🎯 COOL FEATURES

✅ **Station-wise Predictions**: Predict AQI for specific monitoring stations  
✅ **Temporal Analysis**: Any month/year combination  
✅ **Spatial Hotspots**: Identify pollution hotspots in your city  
✅ **Health-Centric**: Disease-specific impact analysis  
✅ **Prevention Guide**: Actionable prevention and mitigation steps  
✅ **Multi-Model Ensemble**: RF + Gradient Boosting for better accuracy  
✅ **Interactive Dashboard**: Charts and statistics for all analysis  
✅ **Regional Coverage**: 25+ Indian cities, 90+ monitoring stations  

---

## 📊 SAMPLE PREDICTIONS

After training, you can predict AQI for:
- **Location**: "Sarojini Market, Delhi"
- **Time**: June 2025  
- **Output**: Predicted AQI ~120 (Moderate Pollution)
- **Recommendation**: Avoid strenuous outdoor activities; use N95 masks

---

## 📁 FILES CREATED/MODIFIED

```
├── aqi_spatial_temporal_model.py ......(new) Main model class
├── aqi_prevention_guide.py ............(new) Prevention recommendations
├── app_enhanced.py ....................(new) Enhanced Streamlit app
├── train_simple.py ....................(new) Quick training script
├── files/
│   ├── station_hour_transformed.csv ...(original)
│   └── station_hour_cleaned.csv .......(new) Cleaned data
├── models/ ............................(new) Trained models directory
├── requirements.txt ...................(updated)
├── SPATIAL_TEMPORAL_SETUP.md ..........(new) Detailed guide
└── QUICK_START.md .....................(this file)
```

---

## ⚡ TIPS

1. **First Time**: Training takes 5-15 min. Be patient!
2. **Predictions**: Use ensemble mode (default) for best results
3. **Safety**: High AQI > 200 means avoid outdoor activities
4. **Weather**: Check forecasts along with AQI predictions
5. **Updates**: Retrain monthly with new data for better accuracy

---

## 🛠️ COMMANDS REFERENCE

```bash
# Install packages
pip install -r requirements.txt

# Train models (run once)
python train_simple.py

# Start Streamlit app
streamlit run app_enhanced.py

# Debug mode
streamlit run app_enhanced.py --logger.level=debug

# Check what's in models directory
ls models/   (Linux/Mac)
dir models   (Windows)
```

---

## ✨ Next Features Coming Soon
- Real-time data integration
- Mobile app version
- Advanced forecasting
- Personal alerts
- Govt integration

---

**Enjoy safer air quality monitoring! 🌍**
