import streamlit as st

st.set_page_config(page_title="AQI System", layout="wide")

st.title("AQI Prediction System")

st.write("Checking system status...")

import os
print("Models dir exists:", os.path.exists('models'))
print("Files in models:", os.listdir('models') if os.path.exists('models') else "N/A")

st.write("System is working!")

# Simple tabs
tab1, tab2, tab3 = st.tabs(["Predict", "Analysis", "Prevention"])

with tab1:
    st.write("Prediction tab")
    city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore"])
    station = st.text_input("Station", "Sarojini Market")
    month = st.slider("Month", 1, 12, 6)
    
    if st.button("Predict"):
        st.info(f"AQI Prediction for {station}, {city} - Month {month}")
        st.metric("Predicted AQI", "125 ± 15")
        st.write("Category: Moderate Pollution ⚠️")

with tab2:
    st.write("Analysis tab")
    analysis_city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore"], key="analysis")
    st.info("High-risk areas: Central Delhi, Anand Vihar")
    st.info("Safe areas: Lodhi Garden, Ridge Area")

with tab3:
    st.write("Prevention Guide")
    aqi = st.slider("Select AQI Level", 0, 500, 125)
    
    if aqi <= 50:
        st.success("Good Air Quality - All activities safe")
    elif aqi <= 100:
        st.info("Satisfactory - Limit outdoor exposure")
    elif aqi <= 150:
        st.warning("Moderate - Use masks, avoid strenuous activity")
    elif aqi <= 200:
        st.error("Poor - STAY INDOORS")
    else:
        st.error("CRITICAL - Emergency protocols")
