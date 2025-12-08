#!/usr/bin/env python3
import csv, math
import matplotlib.pyplot as plt
import numpy as np

def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def pareto(points):
    pts = sorted(points, key=lambda x: (-x[1], x[0]))
    front = []
    best = -math.inf
    for x,y in pts:
        if y > best:
            front.append((x,y))
            best = y
    front.sort()
    return front

def main():
    import os
    src = "evaluation/figures/data/summary_experiments.csv"
    if not os.path.exists(src):
        src = "evaluation/figures/data/scatter.csv"
    rows = read(src)
    # support both summary_experiments.csv and original scatter.csv schema
    x = np.array([float(r.get("latency_increase_pct", r.get("latency", 0.0))) for r in rows])
    y = np.array([float(r.get("percent_saved", r.get("co2_reduction_pct", r.get("co2", 0.0)))) for r in rows])
    c = np.array([float(r.get("cost_savings_pct", 0.0)) for r in rows])
    labels = [r.get("experiment", r.get("config", "run")) for r in rows]
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=80, edgecolors="black")
    for i,l in enumerate(labels):
        ax.annotate(l, (x[i]+0.05, y[i]+0.2), fontsize=9)
    pf = pareto(list(zip(x,y)))
    ax.plot([p[0] for p in pf], [p[1] for p in pf], color="red", linewidth=2)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Cost savings %")
    ax.set_xlabel("Latency increase %")
    ax.set_ylabel("CO₂ reduction %")
    ax.set_title("Multi-Objective Trade-Offs with Pareto Front")
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    out = "evaluation/figures/out/scatter_pareto.png"
    import os
    os.makedirs("evaluation/figures/out", exist_ok=True)
    fig.savefig(out, dpi=160)

if __name__ == "__main__":
    main()
