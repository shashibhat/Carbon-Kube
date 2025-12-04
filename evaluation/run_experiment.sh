#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-baseline}
NAMESPACE=${NAMESPACE:-default}
PROM_URL=${PROM_URL:-http://prometheus-kube-prometheus-prometheus.default.svc:9090}
OUT_DIR=$(cd "$(dirname "$0")" && pwd)/results
mkdir -p "$OUT_DIR"
python3 "$(cd "$(dirname "$0")" && pwd)/generate_dags.py" "$OUT_DIR/manifests" "exp-dag" 3 >/dev/null
for f in "$OUT_DIR"/manifests/*.yaml; do kubectl apply -n "$NAMESPACE" -f "$f"; done
sleep 10
PODS=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
JOULES_JSON=$(curl -s "$PROM_URL/api/v1/query" --data-urlencode 'query=sum by (pod_name,namespace)(kepler_container_joules_total)')
INTENSITY_JSON=$(curl -s "$PROM_URL/api/v1/query" --data-urlencode 'query=carbon_intensity_gco2_per_kwh')
echo "$JOULES_JSON" > "$OUT_DIR/joules.json"
echo "$INTENSITY_JSON" > "$OUT_DIR/intensity.json"
echo '{"workloads":[]}' > "$OUT_DIR/summary.json"
for p in $PODS; do
  TENANT=$(kubectl get pod "$p" -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.carbonkube\.io/tenant}')
  DEADLINE=$(kubectl get pod "$p" -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.carbonkube\.io/deadline}')
  EST=$(kubectl get pod "$p" -n "$NAMESPACE" -o jsonpath='{.metadata.annotations.carbonkube\.io/estimated-runtime-seconds}')
  NODE=$(kubectl get pod "$p" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}')
  ZONE=$(kubectl get node "$NODE" -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
  echo "$p,$TENANT,$DEADLINE,$EST,$ZONE" >> "$OUT_DIR/pods.csv"
done
CRD_JSON=$(kubectl get carbonjobs -n "$NAMESPACE" -o json)
echo "$CRD_JSON" > "$OUT_DIR/carbonjobs.json"
python3 - "$OUT_DIR" "$NAMESPACE" << 'PY'
import sys, json, time
out=sys.argv[1]
ns=sys.argv[2]
with open(out+"/joules.json") as a, open(out+"/intensity.json") as b:
  J=json.load(a); I=json.load(b)
idx={}
for r in I.get("data",{}).get("result",[]):
  m=r.get("metric",{}); v=r.get("value",[0,"0"]) 
  zone=m.get("zone","unknown"); idx[zone]=float(v[1])
pod_joules={}
for r in J.get("data",{}).get("result",[]):
  m=r.get("metric",{}); v=r.get("value",[0,"0"]) 
  pod=m.get("pod_name",""); ns=m.get("namespace","")
  pod_joules[(ns,pod)]=float(v[1])
pods=[]
with open(out+"/pods.csv") as f:
  for line in f:
    name,tenant,deadline,est,node=line.strip().split(",")
    pods.append(dict(name=name,tenant=tenant,deadline=deadline,est=est,node=node))
res={"workloads":[]}
for p in pods:
  z=p["node"] if p["node"]!="" else "unknown"
  ci=idx.get(z,0.0)
  joules=pod_joules.get((ns,p["name"]),0.0)
  kg=joules/3600000.0*ci/1000.0
  dl=p["deadline"]
  est=float(p["est"]) if p["est"] else 0.0
  vio=False
  if dl and est>0:
    try:
      d=time.strptime(dl.replace("Z",""), "%Y-%m-%dT%H:%M:%S")
      dt=time.mktime(d)
      vio=(dt - (time.time()+est))<0
    except: pass
  res["workloads"].append({"pod":p["name"],"tenant":p["tenant"],"region":z,"joules":joules,"intensity_g_per_kwh":ci,"co2_kg":kg,"deadline_violation":vio})
with open(out+"/carbonjobs.json") as cj:
  C=json.load(cj)
  egress=0.0
  for it in C.get("items",[]):
    spec=it.get("spec",{})
    zone="unknown"
    if "dataSources" in spec:
      for ds in spec["dataSources"]:
        r=ds.get("region","unknown")
        ingress=float(ds.get("avgIngressGBPerJob",0))
        read=float(ds.get("avgReadGBPerJob",0))
        if r!=zone:
          egress+=ingress+read
  res["cross_region_egress_gb"]=egress
with open(out+"/summary.json","w") as f:
  json.dump(res,f)
PY
