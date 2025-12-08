#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    rows = read("evaluation/figures/data/sankey.csv")
    flows = []
    labels = []
    orientations = []
    total_in = 0.0
    for r in rows:
        v = float(r["value_kwh"]) if r["value_kwh"] else 0.0
        if r["source"] in ("grid_renewables","grid_fossil") and r["target"]=="cluster_workloads":
            flows.append(v)
            labels.append(r["source"]) 
            orientations.append(1)
            total_in += v
    for r in rows:
        v = float(r["value_kwh"]) if r["value_kwh"] else 0.0
        if r["source"]=="cluster_workloads":
            flows.append(-v)
            labels.append(r["target"]) 
            orientations.append(-1)
    fig = plt.figure(figsize=(8,6))
    Sankey(flows=flows, labels=labels, orientations=orientations).finish()
    plt.title("Energy Flow and Carbon Attribution")
    import os
    os.makedirs("evaluation/figures/out", exist_ok=True)
    plt.savefig("evaluation/figures/out/sankey_energy.png", dpi=160)

if __name__ == "__main__":
    main()
