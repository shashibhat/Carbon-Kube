package emission

// Types that mirror the carbon-kube.proto messages.
// In a real deployment, use generated structs from protoc.

type CarbonScore struct {
    Zone               string  `json:"zone"`
    IntensityGPerKwh   float32 `json:"intensity_g_per_kwh"`
    CpuMultiplier      float32 `json:"cpu_multiplier"`
    ForecastUnixSecond int64   `json:"forecast_unix_second"`
}

type Config struct {
    MigrationThreshold float32  `json:"migration_threshold"`
    GreenZones         []string `json:"green_zones"`
    RLEnabled          bool     `json:"rl_enabled"`
}
