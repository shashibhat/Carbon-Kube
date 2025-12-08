#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt

def read(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    rows = read("evaluation/figures/data/rl.csv")
    ep = [int(r["episode"]) for r in rows]
    rc = [-float(r["reward_carbon"]) for r in rows]
    rk = [-float(r["reward_cost"]) for r in rows]
    rs = [-float(r["reward_sla"]) for r in rows]
    rd = [-float(r["reward_dg"]) for r in rows]
    tot = [float(r["reward_total"]) for r in rows]
    acc = [float(r["validation_accuracy"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(9,5))
    ax1.stackplot(ep, rc, rk, rs, rd, labels=["Carbon","Cost","SLA","Data gravity"], colors=["#2ca02c","#ff7f0e","#1f77b4","#9467bd"])
    ax1.plot(ep, tot, color='black', linewidth=2, label='Total reward')
    ax2 = ax1.twinx()
    ax2.plot(ep, acc, color='gray', linestyle='--', label='Validation accuracy')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative reward components")
    ax2.set_ylabel("Validation accuracy")
    ax1.set_title("RL Convergence and Reward Decomposition")
    lines = ax1.get_lines()+ax2.get_lines()
    labels = [l.get_label() for l in lines]+["Carbon","Cost","SLA","Data gravity"]
    fig.legend(labels, loc='upper left')
    fig.tight_layout()
    import os
    os.makedirs("evaluation/figures/out", exist_ok=True)
    fig.savefig("evaluation/figures/out/rl_stacked_area.png", dpi=160)

if __name__ == "__main__":
    main()
