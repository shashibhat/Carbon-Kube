#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt

def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def series(rows, key, phase):
    return [float(r[key]) for r in rows if r["phase"]==phase]

def main():
    rows = read("evaluation/figures/data/boxplots.csv")
    cats = sorted(set([r["category"] for r in rows]))
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    for i,(metric,label) in enumerate([("latency_increase_pct","Latency %"),("co2_saved_kg","CO₂ saved (kg)"),("sla_violations","SLA violations")]):
        data = []
        xt = []
        for c in cats:
            base = [float(r[metric]) for r in rows if r["category"]==c and r["phase"]=="baseline"]
            carb = [float(r[metric]) for r in rows if r["category"]==c and r["phase"]=="carbon"]
            data.append(base)
            data.append(carb)
            xt.append(c+"\nbaseline")
            xt.append(c+"\ncarbon")
        axes[i].boxplot(data, labels=xt, showmeans=True)
        axes[i].set_title(label)
        axes[i].grid(True, linestyle='--', alpha=0.3)
    fig.suptitle("Performance Metrics Across Workload Types and Phases")
    fig.tight_layout()
    import os
    os.makedirs("evaluation/figures/out", exist_ok=True)
    fig.savefig("evaluation/figures/out/boxplots_metrics.png", dpi=160)

if __name__ == "__main__":
    main()
