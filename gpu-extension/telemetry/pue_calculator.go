package telemetry

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
)

// PUECalculator calculates Power Usage Effectiveness for datacenters
type PUECalculator struct {
	redisClient *redis.Client
	logger      *zap.Logger
	config      *PUEConfig
}

// PUEConfig holds PUE calculation configuration
type PUEConfig struct {
	DefaultPUE          float64                    `json:"defaultPUE"`          // Default PUE value
	UpdateInterval      time.Duration              `json:"updateInterval"`      // How often to update PUE
	DatacenterPUE       map[string]DatacenterInfo  `json:"datacenterPUE"`       // Zone-specific PUE data
	SeasonalAdjustment  bool                       `json:"seasonalAdjustment"`  // Enable seasonal PUE adjustment
	WeatherIntegration  bool                       `json:"weatherIntegration"`  // Enable weather-based PUE
	CoolingEfficiency   map[string]CoolingInfo     `json:"coolingEfficiency"`   // Cooling system efficiency
}

// DatacenterInfo holds datacenter-specific PUE information
type DatacenterInfo struct {
	Zone                string    `json:"zone"`
	BasePUE             float64   `json:"basePUE"`             // Base PUE value
	MinPUE              float64   `json:"minPUE"`              // Minimum achievable PUE
	MaxPUE              float64   `json:"maxPUE"`              // Maximum PUE (worst case)
	CoolingType         string    `json:"coolingType"`         // air, liquid, immersion
	RenewablePercent    float64   `json:"renewablePercent"`    // % of renewable energy
	EfficiencyClass     string    `json:"efficiencyClass"`     // A, B, C, D rating
	LastMeasurement     time.Time `json:"lastMeasurement"`     // Last PUE measurement
	TemperatureRange    [2]float64 `json:"temperatureRange"`   // [min, max] operating temp
	HumidityRange       [2]float64 `json:"humidityRange"`      // [min, max] operating humidity
	AltitudeMeters      float64   `json:"altitudeMeters"`      // Datacenter altitude
	GeographicLocation  Location  `json:"geographicLocation"`  // Lat/lon for weather
}

// CoolingInfo holds cooling system efficiency data
type CoolingInfo struct {
	CoolingType         string  `json:"coolingType"`         // air, liquid, immersion, hybrid
	BaseEfficiency      float64 `json:"baseEfficiency"`      // Base cooling efficiency
	TemperatureCoeff    float64 `json:"temperatureCoeff"`    // Temperature coefficient
	HumidityCoeff       float64 `json:"humidityCoeff"`       // Humidity coefficient
	LoadCoeff           float64 `json:"loadCoeff"`           // Load-dependent coefficient
	SeasonalVariation   float64 `json:"seasonalVariation"`   // Seasonal variation factor
}

// Location represents geographic coordinates
type Location struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

// PUEMeasurement represents a PUE measurement at a point in time
type PUEMeasurement struct {
	Zone                string    `json:"zone"`
	Timestamp           time.Time `json:"timestamp"`
	PUE                 float64   `json:"pue"`
	ITLoad              float64   `json:"itLoad"`              // IT load in kW
	TotalFacilityLoad   float64   `json:"totalFacilityLoad"`   // Total facility load in kW
	CoolingLoad         float64   `json:"coolingLoad"`         // Cooling load in kW
	LightingLoad        float64   `json:"lightingLoad"`        // Lighting load in kW
	UPSLoss             float64   `json:"upsLoss"`             // UPS losses in kW
	PDULoss             float64   `json:"pduLoss"`             // PDU losses in kW
	NetworkLoad         float64   `json:"networkLoad"`         // Network equipment load in kW
	OutsideTemperature  float64   `json:"outsideTemperature"`  // Outside temperature in °C
	OutsideHumidity     float64   `json:"outsideHumidity"`     // Outside humidity in %
	DatacenterTemp      float64   `json:"datacenterTemp"`      // Datacenter temperature in °C
	DatacenterHumidity  float64   `json:"datacenterHumidity"`  // Datacenter humidity in %
	CoolingEfficiency   float64   `json:"coolingEfficiency"`   // Cooling system efficiency
	RenewablePercent    float64   `json:"renewablePercent"`    // % renewable energy used
}

// PUEForecast represents PUE forecast data
type PUEForecast struct {
	Zone                string      `json:"zone"`
	Timestamp           time.Time   `json:"timestamp"`
	ForecastHours       int         `json:"forecastHours"`
	PUEForecast         []float64   `json:"pueForecast"`         // Hourly PUE forecast
	TemperatureForecast []float64   `json:"temperatureForecast"` // Hourly temperature forecast
	LoadForecast        []float64   `json:"loadForecast"`        // Hourly load forecast
	Confidence          []float64   `json:"confidence"`          // Forecast confidence (0-1)
	OptimalWindows      []TimeWindow `json:"optimalWindows"`     // Optimal PUE windows
}

// NewPUECalculator creates a new PUE calculator
func NewPUECalculator(redisClient *redis.Client, logger *zap.Logger) *PUECalculator {
	config := &PUEConfig{
		DefaultPUE:         1.4,
		UpdateInterval:     15 * time.Minute,
		SeasonalAdjustment: true,
		WeatherIntegration: true,
		DatacenterPUE:      getDefaultDatacenterInfo(),
		CoolingEfficiency:  getDefaultCoolingInfo(),
	}

	return &PUECalculator{
		redisClient: redisClient,
		logger:      logger,
		config:      config,
	}
}

// CalculatePUE calculates current PUE for a zone
func (pue *PUECalculator) CalculatePUE(ctx context.Context, zone string, 
	itLoad float64) (*PUEMeasurement, error) {
	
	pue.logger.Debug("Calculating PUE", zap.String("zone", zone), zap.Float64("itLoad", itLoad))

	// Get datacenter info
	dcInfo, exists := pue.config.DatacenterPUE[zone]
	if !exists {
		dcInfo = pue.config.DatacenterPUE["default"]
	}

	measurement := &PUEMeasurement{
		Zone:      zone,
		Timestamp: time.Now(),
		ITLoad:    itLoad,
	}

	// Get weather data if enabled
	var outsideTemp, outsideHumidity float64
	if pue.config.WeatherIntegration {
		weather, err := pue.getWeatherData(ctx, dcInfo.GeographicLocation)
		if err != nil {
			pue.logger.Warn("Failed to get weather data, using defaults", zap.Error(err))
			outsideTemp = 20.0  // Default temperature
			outsideHumidity = 50.0 // Default humidity
		} else {
			outsideTemp = weather.Temperature
			outsideHumidity = weather.Humidity
		}
	} else {
		outsideTemp = 20.0
		outsideHumidity = 50.0
	}

	measurement.OutsideTemperature = outsideTemp
	measurement.OutsideHumidity = outsideHumidity

	// Calculate cooling load based on IT load and environmental conditions
	coolingLoad := pue.calculateCoolingLoad(dcInfo, itLoad, outsideTemp, outsideHumidity)
	measurement.CoolingLoad = coolingLoad

	// Calculate other facility loads
	measurement.LightingLoad = pue.calculateLightingLoad(itLoad)
	measurement.UPSLoss = pue.calculateUPSLoss(itLoad)
	measurement.PDULoss = pue.calculatePDULoss(itLoad)
	measurement.NetworkLoad = pue.calculateNetworkLoad(itLoad)

	// Calculate total facility load
	measurement.TotalFacilityLoad = itLoad + coolingLoad + measurement.LightingLoad + 
		measurement.UPSLoss + measurement.PDULoss + measurement.NetworkLoad

	// Calculate PUE
	if itLoad > 0 {
		measurement.PUE = measurement.TotalFacilityLoad / itLoad
	} else {
		measurement.PUE = dcInfo.BasePUE
	}

	// Apply seasonal adjustment if enabled
	if pue.config.SeasonalAdjustment {
		seasonalFactor := pue.calculateSeasonalFactor(time.Now())
		measurement.PUE *= seasonalFactor
	}

	// Ensure PUE is within reasonable bounds
	measurement.PUE = math.Max(dcInfo.MinPUE, math.Min(dcInfo.MaxPUE, measurement.PUE))

	// Calculate cooling efficiency
	measurement.CoolingEfficiency = pue.calculateCoolingEfficiency(dcInfo, outsideTemp, outsideHumidity, itLoad)

	// Set datacenter conditions (simplified - would be measured in practice)
	measurement.DatacenterTemp = 22.0 + (itLoad/1000.0)*0.5 // Estimate based on load
	measurement.DatacenterHumidity = 45.0 + outsideHumidity*0.1

	// Set renewable percentage
	measurement.RenewablePercent = dcInfo.RenewablePercent

	// Cache the measurement
	if err := pue.cacheMeasurement(ctx, measurement); err != nil {
		pue.logger.Warn("Failed to cache PUE measurement", zap.Error(err))
	}

	return measurement, nil
}

// GetPUE returns the current PUE for a zone (cached or calculated)
func (pue *PUECalculator) GetPUE(ctx context.Context, zone string) (float64, error) {
	// Try to get from cache first
	if pue.redisClient != nil {
		cacheKey := fmt.Sprintf("pue:current:%s", zone)
		cached, err := pue.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			var measurement PUEMeasurement
			if err := json.Unmarshal([]byte(cached), &measurement); err == nil {
				// Check if measurement is recent (within update interval)
				if time.Since(measurement.Timestamp) < pue.config.UpdateInterval {
					return measurement.PUE, nil
				}
			}
		}
	}

	// Calculate new PUE with estimated IT load
	estimatedLoad := pue.estimateITLoad(ctx, zone)
	measurement, err := pue.CalculatePUE(ctx, zone, estimatedLoad)
	if err != nil {
		// Return default PUE if calculation fails
		if dcInfo, exists := pue.config.DatacenterPUE[zone]; exists {
			return dcInfo.BasePUE, nil
		}
		return pue.config.DefaultPUE, nil
	}

	return measurement.PUE, nil
}

// GetPUEForecast returns PUE forecast for a zone
func (pue *PUECalculator) GetPUEForecast(ctx context.Context, zone string, 
	hours int) (*PUEForecast, error) {
	
	// Try to get from cache first
	if pue.redisClient != nil {
		cacheKey := fmt.Sprintf("pue:forecast:%s:%d", zone, hours)
		cached, err := pue.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			var forecast PUEForecast
			if err := json.Unmarshal([]byte(cached), &forecast); err == nil {
				// Check if forecast is recent
				if time.Since(forecast.Timestamp) < 30*time.Minute {
					return &forecast, nil
				}
			}
		}
	}

	// Generate new forecast
	forecast := &PUEForecast{
		Zone:          zone,
		Timestamp:     time.Now(),
		ForecastHours: hours,
		PUEForecast:   make([]float64, hours),
		TemperatureForecast: make([]float64, hours),
		LoadForecast:  make([]float64, hours),
		Confidence:    make([]float64, hours),
	}

	dcInfo, exists := pue.config.DatacenterPUE[zone]
	if !exists {
		dcInfo = pue.config.DatacenterPUE["default"]
	}

	// Generate hourly forecasts
	for i := 0; i < hours; i++ {
		forecastTime := time.Now().Add(time.Duration(i) * time.Hour)
		
		// Forecast temperature (simplified - would use weather API in practice)
		baseTemp := 20.0
		dailyVariation := 10.0 * math.Sin(float64(forecastTime.Hour()-6)*math.Pi/12)
		seasonalVariation := 5.0 * math.Sin(float64(forecastTime.YearDay())*2*math.Pi/365)
		forecastTemp := baseTemp + dailyVariation + seasonalVariation
		forecast.TemperatureForecast[i] = forecastTemp

		// Forecast IT load (simplified - would use historical patterns)
		baseLoad := 1000.0 // kW
		hourlyPattern := 0.8 + 0.4*math.Sin(float64(forecastTime.Hour()-12)*math.Pi/12)
		weeklyPattern := 1.0
		if forecastTime.Weekday() == time.Saturday || forecastTime.Weekday() == time.Sunday {
			weeklyPattern = 0.7 // Lower load on weekends
		}
		forecastLoad := baseLoad * hourlyPattern * weeklyPattern
		forecast.LoadForecast[i] = forecastLoad

		// Calculate PUE for forecasted conditions
		coolingLoad := pue.calculateCoolingLoad(dcInfo, forecastLoad, forecastTemp, 50.0)
		otherLoads := pue.calculateLightingLoad(forecastLoad) + 
			pue.calculateUPSLoss(forecastLoad) + 
			pue.calculatePDULoss(forecastLoad) + 
			pue.calculateNetworkLoad(forecastLoad)
		
		totalLoad := forecastLoad + coolingLoad + otherLoads
		forecastPUE := totalLoad / forecastLoad

		// Apply seasonal adjustment
		if pue.config.SeasonalAdjustment {
			seasonalFactor := pue.calculateSeasonalFactor(forecastTime)
			forecastPUE *= seasonalFactor
		}

		// Ensure within bounds
		forecastPUE = math.Max(dcInfo.MinPUE, math.Min(dcInfo.MaxPUE, forecastPUE))
		forecast.PUEForecast[i] = forecastPUE

		// Calculate confidence (decreases with time)
		confidence := 0.95 - float64(i)*0.05 // 95% confidence decreasing by 5% per hour
		forecast.Confidence[i] = math.Max(0.5, confidence)
	}

	// Find optimal windows (lowest PUE periods)
	forecast.OptimalWindows = pue.findOptimalPUEWindows(forecast.PUEForecast)

	// Cache the forecast
	if pue.redisClient != nil {
		data, _ := json.Marshal(forecast)
		cacheKey := fmt.Sprintf("pue:forecast:%s:%d", zone, hours)
		pue.redisClient.Set(ctx, cacheKey, data, 30*time.Minute)
	}

	return forecast, nil
}

// calculateCoolingLoad calculates cooling load based on IT load and environmental conditions
func (pue *PUECalculator) calculateCoolingLoad(dcInfo DatacenterInfo, itLoad, 
	outsideTemp, outsideHumidity float64) float64 {
	
	coolingInfo, exists := pue.config.CoolingEfficiency[dcInfo.CoolingType]
	if !exists {
		coolingInfo = pue.config.CoolingEfficiency["air"] // Default to air cooling
	}

	// Base cooling load (typically 30-50% of IT load for air cooling)
	baseCoolingRatio := coolingInfo.BaseEfficiency

	// Temperature adjustment
	tempDelta := outsideTemp - 20.0 // Assume 20°C is optimal
	tempAdjustment := 1.0 + (tempDelta * coolingInfo.TemperatureCoeff)

	// Humidity adjustment
	humidityDelta := outsideHumidity - 45.0 // Assume 45% is optimal
	humidityAdjustment := 1.0 + (humidityDelta * coolingInfo.HumidityCoeff)

	// Load adjustment (cooling efficiency decreases at very low or high loads)
	loadFactor := itLoad / 1000.0 // Normalize to 1MW
	loadAdjustment := 1.0 + math.Abs(loadFactor-1.0)*coolingInfo.LoadCoeff

	// Calculate final cooling load
	coolingLoad := itLoad * baseCoolingRatio * tempAdjustment * humidityAdjustment * loadAdjustment

	return math.Max(0.1*itLoad, coolingLoad) // Minimum 10% of IT load
}

// calculateLightingLoad calculates lighting load (typically 1-3% of IT load)
func (pue *PUECalculator) calculateLightingLoad(itLoad float64) float64 {
	return itLoad * 0.02 // 2% of IT load
}

// calculateUPSLoss calculates UPS losses (typically 5-10% of IT load)
func (pue *PUECalculator) calculateUPSLoss(itLoad float64) float64 {
	// UPS efficiency is typically 92-96%
	upsEfficiency := 0.94
	return itLoad * (1.0/upsEfficiency - 1.0)
}

// calculatePDULoss calculates PDU losses (typically 2-5% of IT load)
func (pue *PUECalculator) calculatePDULoss(itLoad float64) float64 {
	return itLoad * 0.03 // 3% of IT load
}

// calculateNetworkLoad calculates network equipment load (typically 3-8% of IT load)
func (pue *PUECalculator) calculateNetworkLoad(itLoad float64) float64 {
	return itLoad * 0.05 // 5% of IT load
}

// calculateSeasonalFactor calculates seasonal PUE adjustment factor
func (pue *PUECalculator) calculateSeasonalFactor(t time.Time) float64 {
	// PUE is typically higher in summer due to increased cooling needs
	dayOfYear := float64(t.YearDay())
	seasonalVariation := 0.1 * math.Sin((dayOfYear-80)*2*math.Pi/365) // Peak in summer
	return 1.0 + seasonalVariation
}

// calculateCoolingEfficiency calculates cooling system efficiency
func (pue *PUECalculator) calculateCoolingEfficiency(dcInfo DatacenterInfo, 
	outsideTemp, outsideHumidity, itLoad float64) float64 {
	
	coolingInfo, exists := pue.config.CoolingEfficiency[dcInfo.CoolingType]
	if !exists {
		return 0.7 // Default efficiency
	}

	baseEfficiency := coolingInfo.BaseEfficiency
	
	// Temperature impact on efficiency
	optimalTemp := 15.0 // Optimal outside temperature for cooling
	tempDelta := math.Abs(outsideTemp - optimalTemp)
	tempImpact := 1.0 - (tempDelta * 0.01) // 1% efficiency loss per degree

	// Load impact on efficiency
	optimalLoad := 0.7 // 70% load is typically optimal
	currentLoad := itLoad / 2000.0 // Assume 2MW capacity
	loadDelta := math.Abs(currentLoad - optimalLoad)
	loadImpact := 1.0 - (loadDelta * 0.1) // 10% efficiency loss per 10% load deviation

	efficiency := baseEfficiency * tempImpact * loadImpact
	return math.Max(0.3, math.Min(0.95, efficiency)) // Bound between 30% and 95%
}

// estimateITLoad estimates current IT load for a zone
func (pue *PUECalculator) estimateITLoad(ctx context.Context, zone string) float64 {
	// In practice, this would query actual power measurements
	// For now, simulate based on time of day and zone
	
	hour := time.Now().Hour()
	baseLoad := 1000.0 // 1MW base load
	
	// Daily pattern (higher during business hours)
	dailyFactor := 0.6 + 0.4*math.Sin(float64(hour-6)*math.Pi/12)
	
	// Zone-specific scaling
	zoneFactors := map[string]float64{
		"us-west1-a":     1.2,
		"us-east1-a":     1.5,
		"europe-west1-a": 0.8,
		"asia-east1-a":   1.0,
		"default":        1.0,
	}
	
	zoneFactor, exists := zoneFactors[zone]
	if !exists {
		zoneFactor = zoneFactors["default"]
	}
	
	return baseLoad * dailyFactor * zoneFactor
}

// getWeatherData gets current weather data for a location
func (pue *PUECalculator) getWeatherData(ctx context.Context, location Location) (*WeatherData, error) {
	// In practice, this would query a weather API
	// For now, simulate weather data
	
	hour := time.Now().Hour()
	dayOfYear := time.Now().YearDay()
	
	// Simulate temperature based on time and location
	baseTemp := 15.0 + location.Latitude*0.3 // Rough latitude adjustment
	dailyVariation := 8.0 * math.Sin(float64(hour-6)*math.Pi/12)
	seasonalVariation := 10.0 * math.Sin(float64(dayOfYear-80)*2*math.Pi/365)
	
	temperature := baseTemp + dailyVariation + seasonalVariation
	humidity := 40.0 + 20.0*math.Sin(float64(hour)*math.Pi/12) // Humidity varies with time
	
	return &WeatherData{
		Temperature: temperature,
		Humidity:    humidity,
		Pressure:    1013.25, // Standard atmospheric pressure
		WindSpeed:   5.0,     // m/s
	}, nil
}

// WeatherData represents weather conditions
type WeatherData struct {
	Temperature float64 `json:"temperature"` // °C
	Humidity    float64 `json:"humidity"`    // %
	Pressure    float64 `json:"pressure"`    // hPa
	WindSpeed   float64 `json:"windSpeed"`   // m/s
}

// findOptimalPUEWindows finds time windows with lowest PUE
func (pue *PUECalculator) findOptimalPUEWindows(pueForecast []float64) []TimeWindow {
	var windows []TimeWindow
	
	if len(pueForecast) < 2 {
		return windows
	}
	
	// Find windows of 2-4 hours with lowest average PUE
	for windowSize := 2; windowSize <= 4 && windowSize <= len(pueForecast); windowSize++ {
		minAvgPUE := math.Inf(1)
		var bestWindow TimeWindow
		
		for start := 0; start <= len(pueForecast)-windowSize; start++ {
			sum := 0.0
			for i := start; i < start+windowSize; i++ {
				sum += pueForecast[i]
			}
			avgPUE := sum / float64(windowSize)
			
			if avgPUE < minAvgPUE {
				minAvgPUE = avgPUE
				startTime := time.Now().Add(time.Duration(start) * time.Hour)
				endTime := startTime.Add(time.Duration(windowSize) * time.Hour)
				
				bestWindow = TimeWindow{
					StartTime:    startTime,
					EndTime:      endTime,
					AvgIntensity: avgPUE,
					Confidence:   0.8, // Simplified confidence
				}
			}
		}
		
		if bestWindow.StartTime != (time.Time{}) {
			windows = append(windows, bestWindow)
		}
	}
	
	return windows
}

// cacheMeasurement caches a PUE measurement
func (pue *PUECalculator) cacheMeasurement(ctx context.Context, measurement *PUEMeasurement) error {
	if pue.redisClient == nil {
		return nil
	}
	
	data, err := json.Marshal(measurement)
	if err != nil {
		return err
	}
	
	// Cache current measurement
	currentKey := fmt.Sprintf("pue:current:%s", measurement.Zone)
	if err := pue.redisClient.Set(ctx, currentKey, data, pue.config.UpdateInterval*2).Err(); err != nil {
		return err
	}
	
	// Store historical measurement
	historyKey := fmt.Sprintf("pue:history:%s:%d", measurement.Zone, measurement.Timestamp.Unix())
	return pue.redisClient.Set(ctx, historyKey, data, 24*time.Hour).Err()
}

// GetPUEHistory returns historical PUE measurements for a zone
func (pue *PUECalculator) GetPUEHistory(ctx context.Context, zone string, 
	hours int) ([]*PUEMeasurement, error) {
	
	if pue.redisClient == nil {
		return nil, fmt.Errorf("Redis client not available")
	}
	
	var measurements []*PUEMeasurement
	now := time.Now()
	
	for i := 0; i < hours; i++ {
		timestamp := now.Add(-time.Duration(i) * time.Hour)
		key := fmt.Sprintf("pue:history:%s:%d", zone, timestamp.Unix())
		
		data, err := pue.redisClient.Get(ctx, key).Result()
		if err != nil {
			continue // Skip missing measurements
		}
		
		var measurement PUEMeasurement
		if err := json.Unmarshal([]byte(data), &measurement); err != nil {
			continue
		}
		
		measurements = append(measurements, &measurement)
	}
	
	return measurements, nil
}

// Default configuration functions

func getDefaultDatacenterInfo() map[string]DatacenterInfo {
	return map[string]DatacenterInfo{
		"us-west1-a": {
			Zone:             "us-west1-a",
			BasePUE:          1.2,
			MinPUE:           1.1,
			MaxPUE:           1.4,
			CoolingType:      "liquid",
			RenewablePercent: 85.0,
			EfficiencyClass:  "A",
			TemperatureRange: [2]float64{18.0, 27.0},
			HumidityRange:    [2]float64{40.0, 60.0},
			AltitudeMeters:   50.0,
			GeographicLocation: Location{
				Latitude:  37.4419,
				Longitude: -122.1430,
			},
		},
		"us-east1-a": {
			Zone:             "us-east1-a",
			BasePUE:          1.4,
			MinPUE:           1.2,
			MaxPUE:           1.7,
			CoolingType:      "air",
			RenewablePercent: 45.0,
			EfficiencyClass:  "B",
			TemperatureRange: [2]float64{20.0, 25.0},
			HumidityRange:    [2]float64{45.0, 55.0},
			AltitudeMeters:   100.0,
			GeographicLocation: Location{
				Latitude:  39.0458,
				Longitude: -76.6413,
			},
		},
		"europe-west1-a": {
			Zone:             "europe-west1-a",
			BasePUE:          1.1,
			MinPUE:           1.05,
			MaxPUE:           1.3,
			CoolingType:      "immersion",
			RenewablePercent: 95.0,
			EfficiencyClass:  "A+",
			TemperatureRange: [2]float64{16.0, 24.0},
			HumidityRange:    [2]float64{40.0, 65.0},
			AltitudeMeters:   20.0,
			GeographicLocation: Location{
				Latitude:  50.8503,
				Longitude: 4.3517,
			},
		},
		"default": {
			Zone:             "default",
			BasePUE:          1.4,
			MinPUE:           1.2,
			MaxPUE:           1.8,
			CoolingType:      "air",
			RenewablePercent: 30.0,
			EfficiencyClass:  "C",
			TemperatureRange: [2]float64{18.0, 26.0},
			HumidityRange:    [2]float64{40.0, 60.0},
			AltitudeMeters:   200.0,
			GeographicLocation: Location{
				Latitude:  40.0,
				Longitude: 0.0,
			},
		},
	}
}

func getDefaultCoolingInfo() map[string]CoolingInfo {
	return map[string]CoolingInfo{
		"air": {
			CoolingType:       "air",
			BaseEfficiency:    0.4,  // 40% of IT load for cooling
			TemperatureCoeff:  0.02, // 2% increase per degree
			HumidityCoeff:     0.005, // 0.5% increase per % humidity
			LoadCoeff:         0.1,  // 10% efficiency loss per 10% load deviation
			SeasonalVariation: 0.15, // 15% seasonal variation
		},
		"liquid": {
			CoolingType:       "liquid",
			BaseEfficiency:    0.25, // 25% of IT load for cooling
			TemperatureCoeff:  0.015,
			HumidityCoeff:     0.002,
			LoadCoeff:         0.05,
			SeasonalVariation: 0.08,
		},
		"immersion": {
			CoolingType:       "immersion",
			BaseEfficiency:    0.15, // 15% of IT load for cooling
			TemperatureCoeff:  0.01,
			HumidityCoeff:     0.001,
			LoadCoeff:         0.03,
			SeasonalVariation: 0.05,
		},
		"hybrid": {
			CoolingType:       "hybrid",
			BaseEfficiency:    0.3,  // 30% of IT load for cooling
			TemperatureCoeff:  0.018,
			HumidityCoeff:     0.003,
			LoadCoeff:         0.07,
			SeasonalVariation: 0.1,
		},
	}
}