#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt

def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    rows = read("evaluation/figures/data/heatmap.csv")
    workloads = [r["workload"] for r in rows]
    regions = [k for k in rows[0].keys() if k.startswith("US-")]
    data = np.array([[float(r[c]) for c in regions] for r in rows])
    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(data, cmap="YlGn")
    ax.set_xticks(np.arange(len(regions)))
    ax.set_yticks(np.arange(len(workloads)))
    ax.set_xticklabels(regions)
    ax.set_yticklabels(workloads)
    for i in range(len(workloads)):
        for j in range(len(regions)):
            ax.text(j, i, f"{int(data[i,j])}", ha="center", va="center", color="black")
    ax.set_title("Regional Scores with Penalties")
    fig.colorbar(im, ax=ax, label="Score (0-100)")
    fig.tight_layout()
    import os
    os.makedirs("evaluation/figures/out", exist_ok=True)
    fig.savefig("evaluation/figures/out/heatmap_scores.png", dpi=160)

if __name__ == "__main__":
    main()
