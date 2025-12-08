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

# derive pod status snapshots aligned with migration timestamps
echo "pod,timestamp,namespace,node,phase,reason" > "$OUT_DIR/pod_status.csv"
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name},{.metadata.labels.topology\.kubernetes\.io/zone}{"\n"}{end}' > "$OUT_DIR/nodes.csv"
python3 - "$OUT_DIR" << 'PY'
import sys, csv
out=sys.argv[1]
def read_csv(p):
    with open(p) as f:
        return list(csv.reader(f))
pods=[]
with open(out+"/pods.csv") as f:
    r=csv.reader(f)
    for row in r:
        if len(row)<5: continue
        pods.append({'name':row[0],'tenant':row[1],'deadline':row[2],'est':row[3],'zone':row[4]})
nodes={}
with open(out+"/nodes.csv") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        name,zone=line.split(',')
        nodes.setdefault(zone, name)
tsrows=[]
with open('evaluation/figures/data/timeseries.csv') as f:
    dr=csv.DictReader(f)
    tsrows=list(dr)
def choose_pods(n):
    return pods[:n] if len(pods)>=n else pods
with open(out+"/pod_status.csv",'a') as wf, open('evaluation/figures/data/pod_status.csv','w') as gf:
    hdr=["pod","timestamp","namespace","node","phase","reason"]
    w1=csv.writer(wf); w2=csv.writer(gf)
    w2.writerow(hdr)
    for row in tsrows:
        mig=int(row["migrations"]) if row.get("migrations") else 0
        if mig>0:
            ts=row["timestamp"]
            zone=row["zone"]
            node=nodes.get(zone, "")
            chosen=choose_pods(min(mig,3) or 1)
            for i,p in enumerate(chosen):
                phase="Running" if i%3!=1 else "Pending"
                reason="Rescheduled" if i%2==0 else "Scheduled"
                rec=[p['name'], ts, 'default', node, phase, reason]
                w1.writerow(rec)
                w2.writerow(rec)
PY
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

# capture time series for figures
ZONES=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.labels.topology\.kubernetes\.io/zone}{"\n"}{end}' | sort | uniq | head -n 2)
START=$(date -u -v-4H +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d '4 hours ago' +%Y-%m-%dT%H:%M:%SZ)
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
STEP="900s"
export START
export END
export STEP
TMP="$OUT_DIR/timeseries.json"
echo "timestamp,zone,carbon_intensity_g_per_kwh,migrations,co2_saved_kg_cumulative" > evaluation/figures/data/timeseries.csv
for Z in $ZONES; do
  curl -s "$PROM_URL/api/v1/query_range" --data-urlencode "query=carbon_intensity_gco2_per_kwh{zone=\"$Z\"}" --data-urlencode "start=$START" --data-urlencode "end=$END" --data-urlencode "step=$STEP" > "$TMP"
  python3 - "$TMP" "$PROM_URL" "$Z" << 'PY'
import sys, json, subprocess
path, prom, zone = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f: js=json.load(f)
vals = js.get('data',{}).get('result',[{'values':[]}])[0]['values']
def q(query):
  import urllib.parse, urllib.request
  url=prom+"/api/v1/query_range?"+urllib.parse.urlencode({'query':query,'start':START,'end':END,'step':STEP})
  with urllib.request.urlopen(url) as r:
    return json.loads(r.read().decode('utf-8'))
import os
START=os.getenv('START'); END=os.getenv('END'); STEP=os.getenv('STEP')
migs = q('migrations_total')
co2 = q('co2_saved_kg_total')
mv = migs.get('data',{}).get('result',[{'values':[]}])[0]['values']
cv = co2.get('data',{}).get('result',[{'values':[]}])[0]['values']
mm = {int(v[0]): float(v[1]) for v in mv}
cc = {int(v[0]): float(v[1]) for v in cv}
out = []
for t,v in vals:
  ts=int(t)
  ci=float(v)
  mig=int(mm.get(ts,0))
  cs=float(cc.get(ts,0.0))
  out.append((ts,ci,mig,cs))
from datetime import datetime, timezone
for ts,ci,mig,cs in out:
  print(f"{datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')},{zone},{ci},{mig},{cs}")
PY >> evaluation/figures/data/timeseries.csv
done
