import streamlit as st
import pandas as pd

st.set_page_config(page_title="AQI Debug", layout="wide")

st.title("AQI Debug Test")
st.write("If you see this, the basic app is working")

try:
    st.write("Loading data...")
    df = pd.read_csv('files/station_hour_cleaned.csv')
    st.success(f"✅ Data loaded: {len(df)} rows")
    
    st.write("Checking columns...")
    st.write(df.columns.tolist())
    
    st.write("Sample data:")
    st.dataframe(df.head())
    
    st.write("Cities:")
    cities = df['City'].unique()
    st.write(list(cities))
    
except Exception as e:
    st.error(f"❌ Error: {e}")
    import traceback
    st.write(traceback.format_exc())
