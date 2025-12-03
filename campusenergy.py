import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# ================= Task 1 - Data Ingestion & Validation ================= #

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

df_combined = pd.DataFrame()

for file in DATA_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(file, on_bad_lines="skip")

        # Adding metadata if missing
        building_name = file.stem.split("_")[0]
        df["building"] = df.get("building", building_name)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError("Timestamp missing in file!")

        df_combined = pd.concat([df_combined, df], ignore_index=True)
        logging.info(f"Loaded: {file.name}")

    except FileNotFoundError:
        logging.error(f"Missing file: {file.name}")
    except Exception as e:
        logging.error(f"Corrupt or invalid file {file.name}: {e}")

# Ensure timestamp indexing
df_combined.set_index("timestamp", inplace=True)
df_combined.sort_index(inplace=True)

# Save initial cleaned combined dataset
df_combined.to_csv(OUTPUT_DIR / "cleaned_energy_data.csv")


# ================= Task 2 - Aggregation Logic ================= #

def calculate_daily_totals(df):
    return df["kwh"].resample("D").sum()

def calculate_weekly_aggregates(df):
    return df["kwh"].resample("W").sum()

def building_wise_summary(df):
    return df.groupby("building")["kwh"].agg(["mean", "min", "max", "sum"])


daily_totals = calculate_daily_totals(df_combined)
weekly_totals = calculate_weekly_aggregates(df_combined)
summary = building_wise_summary(df_combined)
summary.to_csv(OUTPUT_DIR / "building_summary.csv")


# ================= Task 3 - OOP Modeling ================= #

class MeterReading:
    def __init__(self, timestamp, kwh):
        self.timestamp = timestamp
        self.kwh = kwh

class Building:
    def __init__(self, name):
        self.name = name
        self.meter_readings = []

    def add_reading(self, timestamp, kwh):
        self.meter_readings.append(MeterReading(timestamp, kwh))

    def calculate_total_consumption(self):
        return sum(r.kwh for r in self.meter_readings)

    def generate_report(self):
        return {
            "building": self.name,
            "total_consumption": self.calculate_total_consumption(),
        }

class BuildingManager:
    def __init__(self):
        self.buildings = {}

    def add_readings_from_df(self, df):
        for index, row in df.iterrows():
            b = row["building"]
            if b not in self.buildings:
                self.buildings[b] = Building(b)
            self.buildings[b].add_reading(index, row["kwh"])

manager = BuildingManager()
manager.add_readings_from_df(df_combined)


# ================= Task 4 - Visual Dashboard ================= #
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Trend Line Plot
for b_name, df_b in df_combined.groupby("building"):
    axes[0].plot(df_b["kwh"].resample("D").sum(), label=b_name)
axes[0].set_title("Daily Electricity Consumption Trend")
axes[0].set_ylabel("kWh")
axes[0].legend()

# Bar Chart: Weekly Avg
axes[1].bar(summary.index, summary["mean"])
axes[1].set_title("Average Weekly Consumption by Building")
axes[1].set_ylabel("Avg kWh")

# Scatter Plot: Peak Hour Consumption
axes[2].scatter(df_combined.index, df_combined["kwh"])
axes[2].set_title("Peak Hour Consumption")
axes[2].set_ylabel("kWh")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "dashboard.png")


# ================= Task 5 - Executive Summary ================= #

total_campus_kwh = summary["sum"].sum()
highest_building = summary["sum"].idxmax()
peak_time = df_combined["kwh"].idxmax()

summary_text = f"""
EXECUTIVE SUMMARY
=================
Total Campus Consumption: {total_campus_kwh:.2f} kWh
Highest Consuming Building: {highest_building}
Peak Load Time: {peak_time}
Weekly Trend and Daily Consumption visuals saved as dashboard.png
"""

with open(OUTPUT_DIR / "summary.txt", "w") as f:
    f.write(summary_text)

print(summary_text)
print("All tasks completed successfully!")