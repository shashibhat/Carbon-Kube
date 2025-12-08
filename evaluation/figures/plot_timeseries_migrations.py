#!/usr/bin/env python3
import csv, datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def parse(ts):
    return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")

def main():
    import sys, os
    src = sys.argv[1] if len(sys.argv)>1 else "evaluation/figures/data/timeseries.csv"
    outname = sys.argv[2] if len(sys.argv)>2 else "timeseries_migrations.png"
    rows = read(src)
    t = [parse(r["timestamp"]) for r in rows]
    ci = [float(r["carbon_intensity_g_per_kwh"]) for r in rows]
    mig = [int(r["migrations"]) for r in rows]
    co2 = [float(r["co2_saved_kg_cumulative"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax1.plot(t, ci, color='tab:green', label='Carbon intensity (gCO₂/kWh)')
    ax2.plot(t, mig, color='tab:orange', label='Migrations')
    ax3.plot(t, co2, color='tab:blue', label='Cumulative CO₂ saved (kg)')
    ax3.spines["right"].set_position(("axes", 1.1))
    ax1.set_xlabel("Time")
    ax1.set_ylabel("gCO₂/kWh")
    ax2.set_ylabel("Migrations")
    ax3.set_ylabel("CO₂ saved (kg)")
    ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_title("Carbon Intensity, Migrations, and CO₂ Savings")
    labels = [l.get_label() for l in [ax1.lines[0], ax2.lines[0], ax3.lines[0]]]
    fig.legend(labels, loc='upper right')
    fig.tight_layout()
    os.makedirs("evaluation/figures/out", exist_ok=True)
    fig.savefig(os.path.join("evaluation/figures/out", outname), dpi=160)

if __name__ == "__main__":
    main()
