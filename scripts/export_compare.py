import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt

def read_value(path: Path) -> float:
    try:
        data = json.loads(path.read_text())
        res = data["data"]["result"][0]["value"][1]
        return float(res)
    except Exception:
        return 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    d = Path(args.results)

    baseline = {
        "co2_saved_kg_total": read_value(d / "baseline_co2.json"),
        "migrations_total": read_value(d / "baseline_migrations.json"),
        "latency_increase_percent": read_value(d / "baseline_latency.json"),
    }
    carbon = {
        "co2_saved_kg_total": read_value(d / "carbon_co2.json"),
        "migrations_total": read_value(d / "carbon_migrations.json"),
        "latency_increase_percent": read_value(d / "carbon_latency.json"),
    }

    rows = []
    for m in ("co2_saved_kg_total", "migrations_total", "latency_increase_percent"):
        b = baseline.get(m, 0.0)
        c = carbon.get(m, 0.0)
        delta = c - b
        pct = (delta / b * 100.0) if b != 0 else 0.0
        rows.append([m, b, c, delta, pct])

    with open(d / "compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "carbon_aware", "delta", "percent_change"])
        w.writerows(rows)

    labels = [r[0] for r in rows]
    base_vals = [r[1] for r in rows]
    carbon_vals = [r[2] for r in rows]
    x = range(len(labels))
    plt.figure(figsize=(8, 4))
    plt.bar(x, base_vals, width=0.4, label="baseline")
    plt.bar([i + 0.4 for i in x], carbon_vals, width=0.4, label="carbon-aware")
    plt.xticks([i + 0.2 for i in x], labels, rotation=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(d / "compare.png")
    plt.savefig(d / "compare.pdf")

if __name__ == "__main__":
    main()