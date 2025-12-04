package providers

import (
    "encoding/json"
    "fmt"
    "net/http"
    "net/url"
    "time"
)

type ForecastPoint struct {
    Timestamp time.Time
    Value float64
}

type ForecastProvider interface {
    Forecast(region string, start time.Time, end time.Time, step time.Duration) ([]ForecastPoint, error)
}

type PrometheusProvider struct {
    BaseURL string
    Client *http.Client
    Query string
    RegionLabel string
}

func NewPrometheusProvider(baseURL string, query string, regionLabel string) *PrometheusProvider {
    return &PrometheusProvider{BaseURL: baseURL, Client: &http.Client{Timeout: 10 * time.Second}, Query: query, RegionLabel: regionLabel}
}

func (p *PrometheusProvider) Forecast(region string, start time.Time, end time.Time, step time.Duration) ([]ForecastPoint, error) {
    u, err := url.Parse(p.BaseURL)
    if err != nil {
        return nil, err
    }
    u.Path = "/api/v1/query_range"
    params := url.Values{}
    params.Set("query", fmt.Sprintf("%s{%s=\"%s\"}", p.Query, p.RegionLabel, region))
    params.Set("start", start.Format(time.RFC3339))
    params.Set("end", end.Format(time.RFC3339))
    params.Set("step", fmt.Sprintf("%ds", int(step.Seconds())))
    u.RawQuery = params.Encode()
    resp, err := p.Client.Get(u.String())
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    var body struct{
        Status string `json:"status"`
        Data struct{
            ResultType string `json:"resultType"`
            Result []struct{
                Values [][]interface{} `json:"values"`
            } `json:"result"`
        } `json:"data"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
        return nil, err
    }
    out := []ForecastPoint{}
    if len(body.Data.Result) == 0 {
        return out, nil
    }
    for _, v := range body.Data.Result[0].Values {
        ts := int64(v[0].(float64))
        valStr := v[1].(string)
        val, _ := parseFloat(valStr)
        out = append(out, ForecastPoint{Timestamp: time.Unix(ts, 0).UTC(), Value: val})
    }
    return out, nil
}

type ElectricityMapsProvider struct {
    BaseURL string
    Token string
    Client *http.Client
}

func NewElectricityMapsProvider(baseURL string, token string) *ElectricityMapsProvider {
    return &ElectricityMapsProvider{BaseURL: baseURL, Token: token, Client: &http.Client{Timeout: 10 * time.Second}}
}

func (e *ElectricityMapsProvider) Forecast(region string, start time.Time, end time.Time, step time.Duration) ([]ForecastPoint, error) {
    u, err := url.Parse(e.BaseURL)
    if err != nil {
        return nil, err
    }
    u.Path = "/v3/carbon-intensity/forecast"
    params := url.Values{}
    params.Set("zone", region)
    u.RawQuery = params.Encode()
    req, err := http.NewRequest("GET", u.String(), nil)
    if err != nil {
        return nil, err
    }
    req.Header.Set("Authorization", "Bearer "+e.Token)
    resp, err := e.Client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    var body struct{
        Forecast []struct{
            Datetime string `json:"datetime"`
            CarbonIntensity float64 `json:"carbonIntensity"`
        } `json:"forecast"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
        return nil, err
    }
    out := []ForecastPoint{}
    for _, f := range body.Forecast {
        t, err := time.Parse(time.RFC3339, f.Datetime)
        if err != nil {
            continue
        }
        if t.Before(start) || t.After(end) {
            continue
        }
        out = append(out, ForecastPoint{Timestamp: t.UTC(), Value: f.CarbonIntensity})
    }
    return out, nil
}

func parseFloat(s string) (float64, error) {
    var f float64
    err := json.Unmarshal([]byte(s), &f)
    if err != nil {
        return 0, err
    }
    return f, nil
}
