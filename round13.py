import fastf1
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
cache_dir = "f1_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)

# Load 2024 Belgian GP race session
session_2024 = fastf1.get_session(2024, "Belgium", "R")
session_2024.load()

# Extract lap and sector times
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# Convert times to seconds 
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# Group by driver to get average sector times per driver
sector_times_2024 = laps_2024.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# 2025 Qualifying Data Belgian GP
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Lando Norris", "Oscar Piastri", "Charles Leclerc", "Max Verstappen", "Alexander Albon",
        "George Russell", "Yuki Tsunoda", "Isack Hadjar", "Liam Lawson", "Gabriel Bortoleto",
        "Esteban Ocon", "Oliver Bearman", "Pierre Gasly", "Nico Hulkenberg", "Carlos Sainz",
        "Lewis Hamilton", "Franco Colapinto", "Kimi Antonelli", "Fernando Alonso", "Lance Stroll"
    ],
    "QualifyingTime (s)": [
        100.562, 100.647, 100.900, 100.903, 101.201,
        101.260, 101.284, 101.310, 101.328, 102.387,
        101.525, 101.617, 101.633, 101.707, 101.758,
        101.939, 102.022, 102.139, 102.385, 102.502
    ]
})

# Fix driver mapping inconsistencies
driver_mapping = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Isack Hadjar": "HAD", "Kimi Antonelli": "ANT",  
    "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB", "Esteban Ocon": "OCO", "Nico Hulkenberg": "HUL",  
    "Fernando Alonso": "ALO", "Lance Stroll": "STR", "Carlos Sainz": "SAI", "Pierre Gasly": "GAS",  
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO", "Gabriel Bortoleto": "BOR", "Liam Lawson": "LAW",
    "Franco Colapinto": "COL"  # Add missing mapping
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# Merge qualifying data with sector times
merged_data = qualifying_2025.merge(sector_times_2024, left_on="DriverCode", right_on="Driver", how="left")

# Filter to only drivers present in both datasets
common_drivers = set(sector_times_2024['Driver']) & set(merged_data['DriverCode'].dropna())
print(f"Training on {len(common_drivers)} common drivers: {sorted(common_drivers)}")

# Filter training data to only common drivers
training_data = merged_data[merged_data['DriverCode'].isin(common_drivers)].copy()
training_sector_data = sector_times_2024[sector_times_2024['Driver'].isin(common_drivers)].copy()

# Get corresponding lap times for these drivers only
training_lap_times = laps_2024.groupby("Driver")["LapTime (s)"].mean().reset_index()
training_lap_times = training_lap_times[training_lap_times['Driver'].isin(common_drivers)]

# Align the data properly
X_train_data = training_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
y_train_data = training_lap_times.set_index('Driver').loc[training_data['DriverCode']]['LapTime (s)'].values

# Train model on aligned data
X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size=0.2, random_state=1226)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=1226)
model.fit(X_train, y_train)

# For new drivers without 2024 data, use average sector times
avg_sector_times = sector_times_2024[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean()
for col in ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]:
    merged_data[col] = merged_data[col].fillna(avg_sector_times[col])

# Now predict on all 2025 drivers
X_predict = merged_data[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
predicted_race_times = model.predict(X_predict)
qualifying_2025["PredictedRaceTime (s)"] = predicted_race_times

# Rank drivers by predicted race time
qualifying_2025 = qualifying_2025.sort_values(by="PredictedRaceTime (s)")

# Print final predictions
print("\n🏁 Predicted 2025 Belgian GP Winner with New Drivers and Sector Times 🏁\n")
print(qualifying_2025[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")


