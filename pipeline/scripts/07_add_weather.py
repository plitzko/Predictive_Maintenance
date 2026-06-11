"""
07_add_weather.py
Phase 2, Schritt 2: OpenWeatherMap Wetterdaten holen (mit synthetischem Fallback)
Adds temperature_c and precipitation_mm to the dataset.
"""

import pandas as pd
import numpy as np
import os

print("=== Running 07_add_weather.py ===")

# Try to load API key from .env
api_key = None
try:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
except ImportError:
    pass

use_api = False

if api_key:
    # Attempt real API call for historical weather (Munich)
    try:
        import requests
        # OpenWeatherMap historical data requires paid plan for bulk,
        # so we test with a simple current weather call to validate the key
        test_url = f"https://api.openweathermap.org/data/2.5/weather?q=Munich&appid={api_key}&units=metric"
        resp = requests.get(test_url, timeout=10)
        if resp.status_code == 200:
            print("API-Key validiert, aber historische Daten erfordern kostenpflichtigen Plan.")
            print("Verwende synthetischen Fallback mit realistischen Muenchen-Wetterdaten.")
        else:
            print(f"API-Call fehlgeschlagen (Status {resp.status_code}). Verwende Fallback.")
    except Exception as e:
        print(f"API-Call Fehler: {e}. Verwende Fallback.")
else:
    print("Kein OPENWEATHERMAP_API_KEY in .env gefunden. Verwende synthetischen Fallback.")

# Load timestamped data
df = pd.read_csv("data/engine_health/engine_data_with_timestamps.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Loaded {len(df)} rows.")

# Extract date for daily weather join
df["date"] = df["timestamp"].dt.date

# Generate synthetic but realistic Munich weather data
np.random.seed(42)
unique_dates = sorted(df["date"].unique())
n_days = len(unique_dates)

# Temperature: sinusoidal over the year (winter ~0C, summer ~25C) + noise
# Day of year for each date
day_of_year = np.array([(d - unique_dates[0]).days for d in unique_dates])
# Sine curve: minimum around day 0 (Jan 1), maximum around day 182 (Jul 1)
temp_base = 12.5 - 12.5 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
# Add daily noise (+/- 5C)
temp_noise = np.random.normal(0, 3, size=n_days)
temperatures = temp_base + temp_noise
# Clip to realistic Munich range
temperatures = np.clip(temperatures, -15, 38)

# Precipitation: 20% chance per day, if yes: 1-15mm
has_rain = np.random.random(n_days) < 0.20
precipitation = np.where(has_rain, np.random.uniform(1, 15, size=n_days), 0.0)
precipitation = np.round(precipitation, 1)

# Create weather dataframe
weather_df = pd.DataFrame({
    "date": unique_dates,
    "temperature_c": np.round(temperatures, 1),
    "precipitation_mm": precipitation
})

# Convert date column type for merge
df["date"] = pd.to_datetime(df["date"]).dt.date
weather_df["date"] = weather_df["date"]

# Merge
df = df.merge(weather_df, on="date", how="left")

# Drop the helper date column
df.drop(columns=["date"], inplace=True)

# Save
output_path = "data/engine_health/engine_data_enriched.csv"
df.to_csv(output_path, index=False)

# Print summary
print(f"\nMethode: Synthetischer Fallback (realistische Muenchen-Wetterdaten)")
print(f"Temperaturbereich: min={df['temperature_c'].min():.1f}C, "
      f"max={df['temperature_c'].max():.1f}C, "
      f"mean={df['temperature_c'].mean():.1f}C")
rain_days = (df.groupby(df["timestamp"].dt.date)["precipitation_mm"].first() > 0).mean()
print(f"Anteil Tage mit Niederschlag: {rain_days:.1%}")

print("\n=== Done. Outputs saved to data/engine_health/engine_data_enriched.csv ===")
