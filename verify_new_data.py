import pandas as pd

df = pd.read_csv('files/indian_aqi_realistic_2019_2024.csv')
print('✓ Dataset loaded successfully!')
print(f'Shape: {df.shape}')
print(f'\nCities: {df["City"].nunique()}')
print(f'States: {df["State"].nunique()}')
print(f'\nAQI Range: {df["AQI"].min()} - {df["AQI"].max()}')

print('\n' + '='*80)
print('TOP 10 CITIES BY AQI (Most Polluted)')
print('='*80)
top_cities = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10)
for i, (city, aqi) in enumerate(top_cities.items(), 1):
    print(f"{i:2d}. {city:20s} - AQI: {aqi:.1f}")

print('\n' + '='*80)
print('BOTTOM 10 CITIES BY AQI (Cleanest)')
print('='*80)
bottom_cities = df.groupby('City')['AQI'].mean().sort_values(ascending=True).head(10)
for i, (city, aqi) in enumerate(bottom_cities.items(), 1):
    print(f"{i:2d}. {city:20s} - AQI: {aqi:.1f}")

print('\n✓ Dataset verified successfully!')
