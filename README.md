# 🌍 Random Forest AQI Prediction System - Streamlit App

## Quick Start

### Option 1: Run the Streamlit App (Recommended)
```bash
cd "d:/PBL Project sem 6th"
streamlit run app.py
```

This will open the interactive web application in your browser at `http://localhost:8501`

### Option 2: Run the Python Script
```bash
python aqi_prediction_system.py
```

### Option 3: Use the Jupyter Notebook
```bash
jupyter notebook AQI_RandomForest_Prediction.ipynb
```

---

## 📱 Streamlit App Features

### 🏠 Home Page
- Project overview
- Key statistics
- AQI categories guide
- Quick introduction to the system

### 🔮 Predict AQI
- **Interactive Input Form** with sliders for:
  - Pollutant concentrations (PM2.5, PM10, NO2, CO, SO2, O3)
  - Environmental factors (Temperature, Humidity, Wind Speed, Rainfall, Pressure)
  - Urban factors (Vehicle Count, Industrial Activity Index)
- **Instant AQI Prediction** with:
  - Predicted AQI value
  - AQI category (Good → Severe)
  - Health status (SAFE/HAZARDOUS)
  - Specific health effect warnings
  - Recommended precautions
  - Top 5 improvement measures

### 📊 Data Analysis
- AQI category distribution chart
- Pollutant concentration histograms
- City-wise average AQI ranking
- Comprehensive data visualizations

### 📈 Top Recommendations
- Priority-ranked improvement measures
- Feature importance scores
- Implementation guidelines for each pollutant
- Policy recommendations

### ℹ️ About
- Technical details
- Project overview
- Data sources
- Use cases and applications

---

## 🛠️ Installation

### Requirements
- Python 3.8+
- All packages in `requirements.txt`

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

---

## 📊 App Navigation

1. **Sidebar Menu** - Select different pages
2. **Prediction Form** - Enter values and click "Predict AQI"
3. **Results Display** - See health effects and recommendations
4. **Analytics** - View data visualizations and trends

---

## 🎯 How to Use the Prediction Page

1. Enter your **City Name** (for reference)
2. Input the **Pollutant Values**:
   - Use sliders or type values directly
   - Values are pre-filled with typical ranges
3. Click **"Predict AQI"** button
4. View the **Prediction Results**:
   - AQI number
   - Category (with emoji)
   - Health status
   - Health effect warnings
   - Precautions to follow
   - Top improvement measures

---

## 📈 Example Predictions

### Scenario 1: Good Air Quality
- PM2.5: 15, PM10: 30, NO2: 20, CO: 0.5, SO2: 10, O3: 20
- Expected: AQI 0-50 (SAFE)

### Scenario 2: Poor Air Quality
- PM2.5: 85, PM10: 150, NO2: 65, CO: 1.2, SO2: 25, O3: 45
- Expected: AQI 151-200 (HAZARDOUS)

### Scenario 3: Severe Air Quality
- PM2.5: 180, PM10: 280, NO2: 120, CO: 2.5, SO2: 50, O3: 80
- Expected: AQI 300+ (HAZARDOUS)

---

## 🎨 Color Coding

- 🟢 **Green** - Good (0-50) - SAFE
- 🟡 **Yellow** - Satisfactory (51-100) - SAFE
- 🟠 **Orange** - Moderately Polluted (101-150) - MODERATE HAZARD
- 🔴 **Red** - Poor (151-200) - HAZARDOUS
- 🟣 **Purple** - Very Poor (201-300) - HAZARDOUS
- 💀 **Maroon** - Severe (301+) - HAZARDOUS

---

## 📁 Project Structure

```
d:\PBL Project sem 6th\
├── app.py                               (Streamlit app)
├── aqi_prediction_system.py             (Python script)
├── AQI_RandomForest_Prediction.ipynb    (Jupyter notebook)
├── AQI_PREDICTION_GUIDE.md              (User guide)
├── requirements.txt                     (Dependencies)
├── README.md                            (This file)
└── files/
    ├── aqi_cleaned.csv
    ├── indian_aqi_health_impact_2019_2024_cleaned.csv
    └── Indian Urban Air Quality and Health Impact 2019-2024.xlsx
```

---

## 🚀 Performance

- **Model Accuracy:** 44.7% (AQI category classification)
- **Data:** 10,000+ records across 24 Indian cities
- **Training Time:** < 5 seconds
- **Prediction Time:** < 100ms per city
- **Features:** 13 environmental variables

---

## 📚 API Reference

### Making Predictions Programmatically

```python
from aqi_prediction_system import AQIPredictionSystem

system = AQIPredictionSystem()
system.train()

# Single prediction
aqi, category = system.predict_aqi_category(
    pm25=85, pm10=150, no2=65, co=1.2, so2=25, o3=45,
    temp=28, humidity=60, wind=5, rainfall=0, pressure=1010,
    vehicles=150000, industrial=7.5
)

# Display results
system.display_prediction("New Delhi", aqi, category)
```

---

## 🔒 Data Privacy

- Model is trained on aggregated data
- No personal information is collected
- All calculations are done locally
- No data is sent to external servers

---

## 📝 License

This project is for educational and research purposes.

---

## 🤝 Feedback & Support

For questions or improvements, refer to the project documentation or consult the data analysis section in the app.

---

**Last Updated:** February 2026  
**Status:** ✅ Production Ready
