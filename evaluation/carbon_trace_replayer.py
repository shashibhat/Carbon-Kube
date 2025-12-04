#!/usr/bin/env python3
import os, sys, csv, time, requests

def push_series(push_url, job, labels, metric, samples):
    lines = []
    for ts, val in samples:
        lines.append(f"{metric}{labels} {val} {int(ts*1000)}")
    data = "\n".join(lines) + "\n"
    r = requests.put(f"{push_url}/metrics/job/{job}", data=data, headers={"Content-Type":"text/plain"})
    r.raise_for_status()

def main():
    push_url = os.environ.get("PROM_PUSHGATEWAY_URL", "http://prometheus-pushgateway:9091")
    csv_path = sys.argv[1]
    zone = sys.argv[2]
    metric = "carbon_intensity_gco2_per_kwh"
    samples = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("zone") != zone:
                continue
            ts = float(row["timestamp"])
            val = float(row["intensity"])
            samples.append((ts, val))
    labels = f"{{zone=\"{zone}\"}}"
    push_series(push_url, f"carbon_kube_{zone}", labels, metric, samples)

if __name__ == "__main__":
    main()
