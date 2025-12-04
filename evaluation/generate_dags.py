#!/usr/bin/env python3
import os, sys, json, time

def gen_stage(dag_id, stage_id, upstream, runtime_s, cpu_s, deadline_iso, data_sources, policy_ref):
    y = []
    y.append("apiVersion: carbonkube.io/v1")
    y.append("kind: CarbonJobSpec")
    y.append("metadata:")
    y.append("  name: %s-%s" % (dag_id, stage_id))
    y.append("  labels:")
    y.append("    app: %s" % dag_id)
    y.append("spec:")
    y.append("  dagId: %s" % dag_id)
    y.append("  stageId: %s" % stage_id)
    if upstream:
        y.append("  upstreamStages:")
        for u in upstream:
            y.append("    - %s" % u)
    y.append("  estimatedRuntimeSeconds: %d" % runtime_s)
    y.append("  estimatedCpuSeconds: %d" % cpu_s)
    if deadline_iso:
        y.append("  deadline: \"%s\"" % deadline_iso)
    if data_sources:
        y.append("  dataSources:")
        for ds in data_sources:
            y.append("    - type: %s" % ds["type"]) 
            y.append("      resource: %s" % ds["resource"]) 
            y.append("      region: %s" % ds["region"]) 
            if "avgIngressGBPerJob" in ds:
                y.append("      avgIngressGBPerJob: %s" % ds["avgIngressGBPerJob"]) 
            if "avgReadGBPerJob" in ds:
                y.append("      avgReadGBPerJob: %s" % ds["avgReadGBPerJob"]) 
    if policy_ref:
        y.append("  policyRef: \"%s\"" % policy_ref)
    return "\n".join(y) + "\n"

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "./evaluation/manifests"
    os.makedirs(out_dir, exist_ok=True)
    dag_id = sys.argv[2] if len(sys.argv) > 2 else "demo-dag"
    stages = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    deadline = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 6*3600))
    data_sources = [{"type":"kafka","resource":"events","region":"us-west-2","avgIngressGBPerJob":10},{"type":"s3","resource":"s3://bucket","region":"us-east-1","avgReadGBPerJob":20}]
    for i in range(stages):
        stage_id = f"s{i+1}"
        upstream = [] if i==0 else [f"s{i}"]
        doc = gen_stage(dag_id, stage_id, upstream, 900, 1800, deadline, data_sources, "policy-default")
        path = os.path.join(out_dir, f"{dag_id}-{stage_id}.yaml")
        with open(path, "w") as f:
            f.write(doc)
    print(out_dir)

if __name__ == "__main__":
    main()
