import pandas as pd

df = pd.read_csv('files/indian_aqi_health_impact_2019_2024_cleaned.csv')

print("=" * 80)
print(f"Total Records: {len(df)}")
print(f"Unique Cities: {df['City'].nunique()}")
print("\n" + "=" * 80)
print("CITIES AND THEIR AVERAGE AQI VALUES (sorted highest to lowest):")
print("=" * 80)

city_aqi = df.groupby('City')['AQI'].agg(['mean', 'min', 'max', 'count']).sort_values('mean', ascending=False)
city_aqi.columns = ['Avg AQI', 'Min AQI', 'Max AQI', 'Records']
print(city_aqi.to_string())

print("\n" + "=" * 80)
print("DATA CHECK:")
print("=" * 80)
print("\nAll columns in dataset:")
print(df.columns.tolist())
print("\nSample data:")
print(df.head(10))
