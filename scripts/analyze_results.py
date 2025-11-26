#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def extract_prom_value(j):
    """
    Extract numeric value from Prometheus result JSON.
    Format:
      result[0].value = [timestamp, "value"]
    """
    try:
        return float(j["data"]["result"][0]["value"][1])
    except Exception:
        return 0.0

def summarize():
    rows = []

    # -----------------------------
    # Baseline
    # -----------------------------
    base_co2 = extract_prom_value(load_json(RESULTS/"exp1_baseline_co2.json"))
    base_mig = extract_prom_value(load_json(RESULTS/"exp1_baseline_migrations.json"))
    base_lat = extract_prom_value(load_json(RESULTS/"exp1_baseline_latency.json"))

    rows.append({
        "experiment": "baseline",
        "co2": base_co2,
        "migrations": base_mig,
        "latency": base_lat,
    })

    # -----------------------------
    # Carbon-aware
    # -----------------------------
    car_co2 = extract_prom_value(load_json(RESULTS/"exp1_carbon_co2.json"))
    car_mig = extract_prom_value(load_json(RESULTS/"exp1_carbon_migrations.json"))
    car_lat = extract_prom_value(load_json(RESULTS/"exp1_carbon_latency.json"))

    rows.append({
        "experiment": "carbon-aware",
        "co2": car_co2,
        "migrations": car_mig,
        "latency": car_lat,
    })

    df = pd.DataFrame(rows)
    print(df)
    return df

def plot(df):
    plt.figure(figsize=(8,5))
    plt.bar(df["experiment"], df["co2"], label="CO₂ (kg)")
    plt.title("CO₂ Emissions")
    plt.ylabel("kg")
    plt.savefig("results/plot_co2.png")
    print("[OUT] results/plot_co2.png created")

    plt.figure(figsize=(8,5))
    plt.bar(df["experiment"], df["migrations"], label="Migrations")
    plt.title("Pod Migrations")
    plt.ylabel("count")
    plt.savefig("results/plot_migrations.png")
    print("[OUT] results/plot_migrations.png created")

    plt.figure(figsize=(8,5))
    plt.bar(df["experiment"], df["latency"], label="Latency Increase %")
    plt.title("Runtime Latency Impact")
    plt.ylabel("%")
    plt.savefig("results/plot_latency.png")
    print("[OUT] results/plot_latency.png created")

def main():
    df = summarize()
    plot(df)

if __name__ == "__main__":
    main()