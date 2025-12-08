#!/usr/bin/env python3
import csv
import os
import matplotlib.pyplot as plt

def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    rows = read("evaluation/figures/data/sankey.csv")
    src_energy = {"grid_renewables": 0.0, "grid_fossil": 0.0}
    workloads_energy = {}
    workloads_co2 = {}
    avoided = 0.0
    for r in rows:
        v = float(r["value_kwh"]) if r["value_kwh"] else 0.0
        g = float(r["gco2_per_kwh"]) if r["gco2_per_kwh"] else 0.0
        s, t = r["source"], r["target"]
        if s in src_energy and t == "cluster_workloads":
            src_energy[s] += v
        if s == "cluster_workloads":
            workloads_energy[t] = workloads_energy.get(t, 0.0) + v
            co2 = v * g / 1000.0
            workloads_co2[t] = workloads_co2.get(t, 0.0) + co2
        if s == "carbon_kube_savings" and t == "avoided_emissions":
            avoided += v
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Panel A: Energy breakdown
    renew = src_energy.get("grid_renewables", 0.0)
    foss = src_energy.get("grid_fossil", 0.0)
    ax1.bar(["Cluster energy"], [renew], label="Renewables", color="#2ca02c")
    ax1.bar(["Cluster energy"], [foss], bottom=[renew], label="Fossil", color="#d62728")
    wl_names = list(workloads_energy.keys())
    wl_vals = [workloads_energy[n] for n in wl_names]
    ax1.bar(wl_names, wl_vals, color="#1f77b4")
    ax1.set_ylabel("Energy (kWh)")
    ax1.set_title("Energy Breakdown (Sources and Workloads)")
    ax1.tick_params(axis='x', rotation=30)
    ax1.legend(loc="upper left")
    # Panel B: CO2 attribution
    wl_co2_names = list(workloads_co2.keys())
    wl_co2_vals = [workloads_co2[n] for n in wl_co2_names]
    ax2.bar(wl_co2_names, wl_co2_vals, color="#9467bd", label="Workload CO₂ (kg)")
    if avoided > 0:
        ax2.bar(["Avoided"], [avoided], color="#ff7f0e", label="Avoided emissions (kg)")
    ax2.set_ylabel("CO₂ (kg)")
    ax2.set_title("CO₂ Attribution and Avoided Emissions")
    ax2.tick_params(axis='x', rotation=30)
    ax2.legend(loc="upper left")
    fig.suptitle("Energy and Emissions Breakdown")
    fig.tight_layout()
    os.makedirs("evaluation/figures/out", exist_ok=True)
    fig.savefig("evaluation/figures/out/energy_carbon_breakdown.png", dpi=160)

if __name__ == "__main__":
    main()
