package emission

import (
    "net/http"
    "sync/atomic"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
    co2SavedKgTotal = prometheus.NewCounter(
        prometheus.CounterOpts{
            Name: "co2_saved_kg_total",
            Help: "Total CO2 saved in kilograms.",
        },
    )

    migrationsTotal = prometheus.NewCounter(
        prometheus.CounterOpts{
            Name: "migrations_total",
            Help: "Total number of migrations performed.",
        },
    )

    latencyIncreasePercent = prometheus.NewGauge(
        prometheus.GaugeOpts{
            Name: "latency_increase_percent",
            Help: "Approximate latency increase percentage due to carbon-aware scheduling.",
        },
    )

    registered uint32
)

// RegisterMetrics registers and exposes Prometheus metrics on /metrics.
func RegisterMetrics(mux *http.ServeMux) {
    if atomic.CompareAndSwapUint32(&registered, 0, 1) {
        prometheus.MustRegister(co2SavedKgTotal, migrationsTotal, latencyIncreasePercent)
    }
    mux.Handle("/metrics", promhttp.Handler())
}

// RecordCO2Saved increments the CO2 saved counter.
func RecordCO2Saved(kg float64) {
    co2SavedKgTotal.Add(kg)
}

// RecordMigration increments the migration counter.
func RecordMigration() {
    migrationsTotal.Inc()
}

// SetLatencyIncrease sets the latency increase gauge.
func SetLatencyIncrease(percent float64) {
    latencyIncreasePercent.Set(percent)
}
