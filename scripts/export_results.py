import argparse
import csv
import os
import time
import requests
import json
from pathlib import Path
import matplotlib.pyplot as plt

def query_prom(q, base="http://localhost:9090"):
    r = requests.get(f"{base}/api/v1/query", params={"query": q}, timeout=10)
    r.raise_for_status()
    return r.json()

def write_csv(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prom", default="http://localhost:9090")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    co2 = query_prom("co2_saved_kg_total", args.prom)
    mig = query_prom("migrations_total", args.prom)
    lat = query_prom("latency_increase_percent", args.prom)

    def extract(v):
        try:
            res = v["data"]["result"][0]["value"]
            ts = int(res[0])
            val = float(res[1])
            return ts, val
        except Exception:
            return int(time.time()), 0.0

    rows = []
    ts1, co2v = extract(co2)
    ts2, migv = extract(mig)
    ts3, latv = extract(lat)
    rows.append([ts1, co2v, migv, latv])
    write_csv(out / "metrics.csv", rows, ["timestamp","co2_saved_kg_total","migrations_total","latency_increase_percent"])

    plt.figure(figsize=(6,4))
    plt.bar(["CO2","Mig","Lat"],[co2v, migv, latv])
    plt.tight_layout()
    plt.savefig(out / "metrics.png")

    with open(out / "metrics.json","w") as f:
        json.dump({"co2_saved_kg_total": co2v, "migrations_total": migv, "latency_increase_percent": latv}, f)

if __name__ == "__main__":
    main()