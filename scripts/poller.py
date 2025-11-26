#!/usr/bin/env python3
"""
Carbon Intensity Poller for Carbon-Kube

This script polls carbon intensity data from multiple sources:
1. Electricity Maps API (primary)
2. NOAA Weather API (fallback for renewable energy proxy)
3. AWS Carbon Footprint API (regional data)

The data is cached in a Kubernetes ConfigMap for use by the scheduler plugin.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from retrying import retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_MAP_NAME = "carbon-scores"
CONFIG_MAP_NAMESPACE = "default"
DEFAULT_INTENSITY = 200.0  # gCO2/kWh
CACHE_DURATION = 600  # 10 minutes in seconds

# Zone mappings: AWS regions to Electricity Maps zones
ZONE_MAPPINGS = {
    "us-west-1": "US-CAL-CISO",      # California - typically green
    "us-west-2": "US-NW-PACW",       # Pacific Northwest - very green (hydro)
    "us-east-1": "US-VA-NOR",        # Virginia - mixed
    "us-east-2": "US-MIDW-MISO",     # Ohio - coal heavy
    "eu-west-1": "IE",               # Ireland - wind heavy
    "eu-central-1": "DE",            # Germany - mixed renewable
    "ap-southeast-1": "SG",          # Singapore - gas heavy
    "ap-northeast-1": "JP-TK",       # Tokyo - mixed
}

# NOAA weather stations for renewable energy proxy
NOAA_STATIONS = {
    "us-west-2": {"lat": 47.6, "lon": -122.3, "station": "KSEA"},  # Seattle
    "us-west-1": {"lat": 37.8, "lon": -122.4, "station": "KSFO"}, # San Francisco
    "us-east-1": {"lat": 38.9, "lon": -77.0, "station": "KDCA"},  # Washington DC
    "us-east-2": {"lat": 39.1, "lon": -84.5, "station": "KCVG"},  # Cincinnati
}


class CarbonIntensityPoller:
    """Main poller class for carbon intensity data."""
    
    def __init__(self):
        self.electricity_maps_api_key = os.getenv('ELECTRICITY_MAPS_API_KEY')
        self.noaa_api_key = os.getenv('NOAA_API_KEY')  # Optional
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Carbon-Kube/1.0.0'
        })
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except config.ConfigException:
            try:
                config.load_kube_config()
            except config.ConfigException:
                logger.error("Could not load Kubernetes configuration")
                sys.exit(1)
        
        self.k8s_client = client.CoreV1Api()
        logger.info("Carbon Intensity Poller initialized")

    @retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000)
    def fetch_electricity_maps_data(self, zone: str) -> Optional[Dict]:
        """Fetch carbon intensity from Electricity Maps API."""
        if not self.electricity_maps_api_key:
            logger.warning("Electricity Maps API key not provided")
            return None
        
        url = f"https://api.electricitymaps.com/v3/carbon-intensity/latest"
        params = {"zone": zone}
        headers = {"auth-token": self.electricity_maps_api_key}
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched Electricity Maps data for {zone}: {data.get('carbonIntensity')} gCO2/kWh")
            return {
                "intensity": data.get("carbonIntensity", DEFAULT_INTENSITY),
                "timestamp": int(time.time()),
                "source": "electricity_maps",
                "forecast": self._fetch_forecast(zone)
            }
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Electricity Maps data for {zone}: {e}")
            return None

    def _fetch_forecast(self, zone: str) -> List[float]:
        """Fetch 24-hour forecast from Electricity Maps."""
        if not self.electricity_maps_api_key:
            return []
        
        url = f"https://api.electricitymaps.com/v3/carbon-intensity/forecast"
        params = {"zone": zone}
        headers = {"auth-token": self.electricity_maps_api_key}
        
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecast = []
            for item in data.get("forecast", [])[:24]:  # Next 24 hours
                forecast.append(item.get("carbonIntensity", DEFAULT_INTENSITY))
            
            return forecast
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch forecast for {zone}: {e}")
            return []

    @retry(stop_max_attempt_number=2, wait_fixed=2000)
    def fetch_noaa_renewable_proxy(self, aws_region: str) -> Optional[Dict]:
        """
        Fetch renewable energy proxy data from NOAA weather API.
        Uses solar irradiance and wind speed as proxies for renewable generation.
        """
        station_info = NOAA_STATIONS.get(aws_region)
        if not station_info:
            return None
        
        # NOAA API endpoint for current conditions
        url = f"https://api.weather.gov/stations/{station_info['station']}/observations/latest"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            properties = data.get("properties", {})
            
            # Extract renewable energy indicators
            solar_radiation = properties.get("solarRadiation", {}).get("value", 0) or 0
            wind_speed = properties.get("windSpeed", {}).get("value", 0) or 0
            cloud_layers = properties.get("cloudLayers", [])
            
            # Calculate renewable energy score (0-1)
            renewable_score = self._calculate_renewable_score(
                solar_radiation, wind_speed, cloud_layers
            )
            
            # Convert to carbon intensity (inverse relationship)
            # High renewable score = low carbon intensity
            base_intensity = 400.0  # Base grid intensity
            intensity = base_intensity * (1.0 - renewable_score * 0.6)  # Up to 60% reduction
            
            logger.info(f"NOAA renewable proxy for {aws_region}: {intensity:.1f} gCO2/kWh (score: {renewable_score:.2f})")
            
            return {
                "intensity": intensity,
                "timestamp": int(time.time()),
                "source": "noaa_proxy",
                "renewable_score": renewable_score,
                "solar_radiation": solar_radiation,
                "wind_speed": wind_speed
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch NOAA data for {aws_region}: {e}")
            return None

    def _calculate_renewable_score(self, solar_radiation: float, wind_speed: float, cloud_layers: List) -> float:
        """Calculate renewable energy score from weather data."""
        score = 0.0
        
        # Solar component (0-0.5)
        if solar_radiation > 0:
            # Normalize solar radiation (typical max ~1000 W/m²)
            solar_score = min(solar_radiation / 1000.0, 1.0) * 0.5
            
            # Adjust for cloud cover
            if cloud_layers:
                avg_coverage = sum(layer.get("amount", {}).get("value", 0) or 0 for layer in cloud_layers) / len(cloud_layers)
                cloud_factor = 1.0 - (avg_coverage / 100.0)  # Reduce score for clouds
                solar_score *= cloud_factor
            
            score += solar_score
        
        # Wind component (0-0.5)
        if wind_speed > 0:
            # Normalize wind speed (optimal ~10-15 m/s for wind turbines)
            if wind_speed <= 15:
                wind_score = (wind_speed / 15.0) * 0.5
            else:
                # Too windy (turbines shut down)
                wind_score = max(0, (30 - wind_speed) / 15.0) * 0.5
            
            score += wind_score
        
        return min(score, 1.0)

    def fetch_aws_carbon_data(self, region: str) -> Optional[Dict]:
        """
        Fetch AWS-specific carbon data if available.
        This is a placeholder for future AWS Carbon Footprint API integration.
        """
        # AWS doesn't provide real-time carbon intensity API yet
        # This would integrate with AWS Carbon Footprint tool data
        logger.debug(f"AWS carbon data not available for {region}")
        return None

    def get_zone_intensity(self, aws_region: str) -> Dict:
        """Get carbon intensity for an AWS region using multiple sources."""
        electricity_maps_zone = ZONE_MAPPINGS.get(aws_region)
        
        # Try Electricity Maps first (most accurate)
        if electricity_maps_zone:
            data = self.fetch_electricity_maps_data(electricity_maps_zone)
            if data:
                return {
                    "zone": aws_region,
                    "intensity": data["intensity"],
                    "timestamp": data["timestamp"],
                    "forecast": data.get("forecast", []),
                    "source": "electricity_maps"
                }
        
        # Fallback to NOAA renewable proxy
        noaa_data = self.fetch_noaa_renewable_proxy(aws_region)
        if noaa_data:
            return {
                "zone": aws_region,
                "intensity": noaa_data["intensity"],
                "timestamp": noaa_data["timestamp"],
                "forecast": [],
                "source": "noaa_proxy",
                "renewable_score": noaa_data.get("renewable_score", 0)
            }
        
        # Final fallback to regional defaults
        regional_defaults = {
            "us-west-1": 150.0,  # California - renewable heavy
            "us-west-2": 120.0,  # Pacific Northwest - hydro heavy
            "us-east-1": 250.0,  # Virginia - mixed
            "us-east-2": 350.0,  # Ohio - coal heavy
            "eu-west-1": 180.0,  # Ireland - wind
            "eu-central-1": 220.0,  # Germany - mixed
        }
        
        default_intensity = regional_defaults.get(aws_region, DEFAULT_INTENSITY)
        logger.warning(f"Using default intensity for {aws_region}: {default_intensity} gCO2/kWh")
        
        return {
            "zone": aws_region,
            "intensity": default_intensity,
            "timestamp": int(time.time()),
            "forecast": [],
            "source": "default"
        }

    def poll_all_zones(self) -> Dict[str, Dict]:
        """Poll carbon intensity for all configured zones."""
        results = {}
        
        for aws_region in ZONE_MAPPINGS.keys():
            try:
                zone_data = self.get_zone_intensity(aws_region)
                results[aws_region] = zone_data
                logger.info(f"Polled {aws_region}: {zone_data['intensity']:.1f} gCO2/kWh ({zone_data['source']})")
            except Exception as e:
                logger.error(f"Failed to poll {aws_region}: {e}")
                # Add fallback data
                results[aws_region] = {
                    "zone": aws_region,
                    "intensity": DEFAULT_INTENSITY,
                    "timestamp": int(time.time()),
                    "forecast": [],
                    "source": "error_fallback"
                }
        
        return results

    def update_configmap(self, zone_data: Dict[str, Dict]) -> bool:
        """Update Kubernetes ConfigMap with carbon intensity data."""
        try:
            # Prepare ConfigMap data
            configmap_data = {
                "zones": json.dumps(zone_data, indent=2),
                "last_updated": datetime.utcnow().isoformat(),
                "threshold": "200.0"  # Default threshold, can be updated by RL tuner
            }
            
            # Try to get existing ConfigMap
            try:
                existing_cm = self.k8s_client.read_namespaced_config_map(
                    name=CONFIG_MAP_NAME,
                    namespace=CONFIG_MAP_NAMESPACE
                )
                
                # Preserve existing threshold if set
                if existing_cm.data and "threshold" in existing_cm.data:
                    configmap_data["threshold"] = existing_cm.data["threshold"]
                
                # Update existing ConfigMap
                existing_cm.data = configmap_data
                self.k8s_client.patch_namespaced_config_map(
                    name=CONFIG_MAP_NAME,
                    namespace=CONFIG_MAP_NAMESPACE,
                    body=existing_cm
                )
                logger.info("Updated existing ConfigMap")
                
            except ApiException as e:
                if e.status == 404:
                    # Create new ConfigMap
                    configmap = client.V1ConfigMap(
                        metadata=client.V1ObjectMeta(
                            name=CONFIG_MAP_NAME,
                            namespace=CONFIG_MAP_NAMESPACE
                        ),
                        data=configmap_data
                    )
                    
                    self.k8s_client.create_namespaced_config_map(
                        namespace=CONFIG_MAP_NAMESPACE,
                        body=configmap
                    )
                    logger.info("Created new ConfigMap")
                else:
                    raise
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update ConfigMap: {e}")
            return False

    def run_once(self) -> bool:
        """Run one polling cycle."""
        logger.info("Starting carbon intensity polling cycle")
        
        try:
            # Poll all zones
            zone_data = self.poll_all_zones()
            
            # Update ConfigMap
            success = self.update_configmap(zone_data)
            
            if success:
                logger.info(f"Polling cycle completed successfully. Updated {len(zone_data)} zones.")
            else:
                logger.error("Polling cycle failed to update ConfigMap")
            
            return success
            
        except Exception as e:
            logger.error(f"Polling cycle failed: {e}")
            return False


def main():
    """Main entry point."""
    poller = CarbonIntensityPoller()
    
    # Check if running as one-shot or continuous
    run_once = os.getenv('RUN_ONCE', 'true').lower() == 'true'
    
    if run_once:
        # Single execution (for CronJob)
        success = poller.run_once()
        sys.exit(0 if success else 1)
    else:
        # Continuous execution (for Deployment)
        interval = int(os.getenv('POLL_INTERVAL', '300'))  # 5 minutes default
        
        logger.info(f"Starting continuous polling every {interval} seconds")
        
        while True:
            try:
                poller.run_once()
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                time.sleep(60)  # Wait before retrying


if __name__ == "__main__":
    main()