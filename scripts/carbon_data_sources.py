#!/usr/bin/env python3
"""
Carbon Data Sources Module

This module provides utilities for fetching carbon intensity data from various sources.
It's used by the poller.py script.
"""

from typing import Dict, Optional
import requests
import logging

logger = logging.getLogger(__name__)


class ElectricityMapsClient:
    """Client for Electricity Maps API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.electricitymaps.com/v3"
        self.session = requests.Session()
        self.session.headers.update({
            "auth-token": api_key,
            "User-Agent": "Carbon-Kube/1.0.0"
        })
    
    def get_latest_carbon_intensity(self, zone: str) -> Optional[Dict]:
        """Get latest carbon intensity for a zone."""
        url = f"{self.base_url}/carbon-intensity/latest"
        params = {"zone": zone}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch from Electricity Maps: {e}")
            return None
    
    def get_forecast(self, zone: str) -> Optional[Dict]:
        """Get carbon intensity forecast for a zone."""
        url = f"{self.base_url}/carbon-intensity/forecast"
        params = {"zone": zone}
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch forecast from Electricity Maps: {e}")
            return None


class NOAAClient:
    """Client for NOAA Weather API (used as renewable energy proxy)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.weather.gov"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Carbon-Kube/1.0.0"
        })
    
    def get_station_observations(self, station_id: str) -> Optional[Dict]:
        """Get latest observations from a weather station."""
        url = f"{self.base_url}/stations/{station_id}/observations/latest"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch from NOAA: {e}")
            return None
    
    def calculate_renewable_score(self, observations: Dict) -> float:
        """
        Calculate renewable energy score based on weather observations.
        Higher score indicates better renewable energy conditions.
        """
        if not observations or "properties" not in observations:
            return 0.5  # Neutral score
        
        props = observations["properties"]
        score = 0.0
        
        # Solar component (0-0.5)
        if "skyCover" in props:
            sky_cover = props.get("skyCover", {}).get("value", 50)
            # Lower sky cover = more solar
            solar_score = (1.0 - (sky_cover / 100.0)) * 0.5
            score += solar_score
        
        # Wind component (0-0.5)
        if "windSpeed" in props:
            wind_speed = props.get("windSpeed", {}).get("value", 0)
            # Optimal wind speed ~10-15 m/s
            if 0 < wind_speed <= 15:
                wind_score = (wind_speed / 15.0) * 0.5
            elif wind_speed > 15:
                # Too windy (turbines shut down)
                wind_score = max(0, (30 - wind_speed) / 15.0) * 0.5
            else:
                wind_score = 0.0
            score += wind_score
        
        return min(score, 1.0)


class AWSCarbonClient:
    """Client for AWS Carbon Footprint data."""
    
    def __init__(self):
        # AWS Carbon Footprint API integration would go here
        # Currently, AWS doesn't provide a public API for this
        pass
    
    def get_region_carbon_data(self, region: str) -> Optional[Dict]:
        """Get carbon data for an AWS region."""
        # Placeholder for future AWS Carbon Footprint API integration
        logger.debug(f"AWS carbon data not available for {region}")
        return None


