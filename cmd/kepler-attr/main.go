package main

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
    "net/url"
    "os"
)

func promQuery(base string, q string) (string, error) {
    u, err := url.Parse(base)
    if err != nil { return "", err }
    u.Path = "/api/v1/query"
    params := url.Values{}
    params.Set("query", q)
    u.RawQuery = params.Encode()
    resp, err := http.DefaultClient.Get(u.String())
    if err != nil { return "", err }
    defer resp.Body.Close()
    var body struct{
        Status string `json:"status"`
        Data struct{
            ResultType string `json:"resultType"`
            Result []struct{
                Metric map[string]string `json:"metric"`
                Value []interface{} `json:"value"`
            } `json:"result"`
        } `json:"data"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&body); err != nil { return "", err }
    out := ""
    for _, r := range body.Data.Result {
        val := r.Value[1].(string)
        out += fmt.Sprintf("%s %s\n", r.Metric["__name__"], val)
    }
    return out, nil
}

func metricsHandler(w http.ResponseWriter, r *http.Request) {
    base := os.Getenv("CARBONKUBE_PROMETHEUS_URL")
    w.Header().Set("Content-Type", "text/plain; version=0.0.4")
    s1, _ := promQuery(base, "sum by (tenant)(label_replace(kepler_container_joules_total,\"tenant\",\"$1\",\"pod_name\",\"(.*)\"))")
    s2, _ := promQuery(base, "sum by (tenant)(label_replace(kepler_container_joules_total,\"tenant\",\"$1\",\"pod_name\",\"(.*)\"))")
    s3, _ := promQuery(base, "sum by (namespace)(kepler_container_joules_total)")
    fmt.Fprintf(w, "%s", s1)
    fmt.Fprintf(w, "%s", s2)
    fmt.Fprintf(w, "%s", s3)
}

func main() {
    _ = context.Background()
    http.HandleFunc("/metrics", metricsHandler)
    _ = http.ListenAndServe(":8080", nil)
}
