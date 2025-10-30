#!/usr/bin/env python3
"""
Unit tests for the Carbon Intensity Poller
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import time
from datetime import datetime, timezone
import requests
from poller import CarbonIntensityPoller


class TestCarbonIntensityPoller(unittest.TestCase):
    """Test cases for CarbonIntensityPoller"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'electricity_maps': {
                'api_key': 'test_key',
                'base_url': 'https://api.electricitymap.org/v3',
                'zones': ['US-CA', 'US-TX']
            },
            'noaa': {
                'api_key': 'noaa_test_key',
                'base_url': 'https://api.weather.gov'
            },
            'aws': {
                'enabled': False
            },
            'output': {
                'configmap_name': 'carbon-intensity-data',
                'configmap_namespace': 'carbon-kube'
            },
            'cache_ttl': 300,
            'retry_attempts': 3,
            'retry_delay': 1
        }
        self.poller = CarbonIntensityPoller(self.config)

    @patch('requests.get')
    def test_fetch_electricity_maps_success(self, mock_get):
        """Test successful fetch from Electricity Maps API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'carbonIntensity': 250,
            'datetime': '2024-01-01T12:00:00Z',
            'zone': 'US-CA'
        }
        mock_get.return_value = mock_response

        result = self.poller._fetch_electricity_maps('US-CA')
        
        self.assertEqual(result['carbon_intensity'], 250)
        self.assertEqual(result['zone'], 'US-CA')
        self.assertIn('timestamp', result)

    @patch('requests.get')
    def test_fetch_electricity_maps_failure(self, mock_get):
        """Test failure handling for Electricity Maps API"""
        mock_get.side_effect = requests.RequestException("API Error")
        
        result = self.poller._fetch_electricity_maps('US-CA')
        
        self.assertIsNone(result)

    @patch('requests.get')
    def test_fetch_noaa_weather_success(self, mock_get):
        """Test successful fetch from NOAA API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'properties': {
                'periods': [{
                    'temperature': 75,
                    'windSpeed': '10 mph',
                    'shortForecast': 'Sunny'
                }]
            }
        }
        mock_get.return_value = mock_response

        result = self.poller._fetch_noaa_weather(37.7749, -122.4194)
        
        self.assertEqual(result['temperature'], 75)
        self.assertEqual(result['wind_speed'], '10 mph')
        self.assertEqual(result['conditions'], 'Sunny')

    @patch('requests.get')
    def test_fetch_noaa_weather_failure(self, mock_get):
        """Test failure handling for NOAA API"""
        mock_get.side_effect = requests.RequestException("API Error")
        
        result = self.poller._fetch_noaa_weather(37.7749, -122.4194)
        
        self.assertIsNone(result)

    def test_calculate_renewable_proxy(self):
        """Test renewable energy proxy calculation"""
        weather_data = {
            'temperature': 75,
            'wind_speed': '15 mph',
            'conditions': 'Sunny'
        }
        
        proxy = self.poller._calculate_renewable_proxy(weather_data)
        
        self.assertIsInstance(proxy, float)
        self.assertGreaterEqual(proxy, 0)
        self.assertLessEqual(proxy, 100)

    def test_estimate_carbon_from_weather(self):
        """Test carbon intensity estimation from weather data"""
        weather_data = {
            'temperature': 75,
            'wind_speed': '15 mph',
            'conditions': 'Sunny'
        }
        
        carbon_intensity = self.poller._estimate_carbon_from_weather(weather_data)
        
        self.assertIsInstance(carbon_intensity, float)
        self.assertGreater(carbon_intensity, 0)

    @patch('kubernetes.client.CoreV1Api')
    def test_update_configmap_success(self, mock_k8s_api):
        """Test successful ConfigMap update"""
        mock_api_instance = Mock()
        mock_k8s_api.return_value = mock_api_instance
        
        data = {'US-CA': {'carbon_intensity': 250}}
        
        # Test creation (ConfigMap doesn't exist)
        mock_api_instance.read_namespaced_config_map.side_effect = Exception("Not found")
        
        result = self.poller._update_configmap(data)
        
        self.assertTrue(result)
        mock_api_instance.create_namespaced_config_map.assert_called_once()

    @patch('kubernetes.client.CoreV1Api')
    def test_update_configmap_update(self, mock_k8s_api):
        """Test ConfigMap update when it already exists"""
        mock_api_instance = Mock()
        mock_k8s_api.return_value = mock_api_instance
        
        # Mock existing ConfigMap
        existing_cm = Mock()
        existing_cm.data = {'carbon-intensity': '{}'}
        mock_api_instance.read_namespaced_config_map.return_value = existing_cm
        
        data = {'US-CA': {'carbon_intensity': 250}}
        
        result = self.poller._update_configmap(data)
        
        self.assertTrue(result)
        mock_api_instance.patch_namespaced_config_map.assert_called_once()

    def test_cache_functionality(self):
        """Test caching mechanism"""
        # Test cache miss
        self.assertIsNone(self.poller._get_cached_data('US-CA'))
        
        # Test cache set and hit
        data = {'carbon_intensity': 250, 'timestamp': time.time()}
        self.poller._cache_data('US-CA', data)
        
        cached = self.poller._get_cached_data('US-CA')
        self.assertIsNotNone(cached)
        self.assertEqual(cached['carbon_intensity'], 250)

    def test_cache_expiry(self):
        """Test cache expiry functionality"""
        # Set data with old timestamp
        old_data = {'carbon_intensity': 250, 'timestamp': time.time() - 400}
        self.poller._cache_data('US-CA', old_data)
        
        # Should return None due to expiry
        cached = self.poller._get_cached_data('US-CA')
        self.assertIsNone(cached)

    @patch.object(CarbonIntensityPoller, '_fetch_electricity_maps')
    @patch.object(CarbonIntensityPoller, '_update_configmap')
    def test_poll_once_success(self, mock_update_cm, mock_fetch_em):
        """Test single poll execution"""
        mock_fetch_em.return_value = {
            'carbon_intensity': 250,
            'zone': 'US-CA',
            'timestamp': time.time()
        }
        mock_update_cm.return_value = True
        
        result = self.poller.poll_once()
        
        self.assertTrue(result)
        mock_fetch_em.assert_called()
        mock_update_cm.assert_called_once()

    @patch.object(CarbonIntensityPoller, '_fetch_electricity_maps')
    @patch.object(CarbonIntensityPoller, '_fetch_noaa_weather')
    @patch.object(CarbonIntensityPoller, '_update_configmap')
    def test_poll_once_fallback_to_noaa(self, mock_update_cm, mock_fetch_noaa, mock_fetch_em):
        """Test fallback to NOAA when Electricity Maps fails"""
        mock_fetch_em.return_value = None
        mock_fetch_noaa.return_value = {
            'temperature': 75,
            'wind_speed': '15 mph',
            'conditions': 'Sunny'
        }
        mock_update_cm.return_value = True
        
        result = self.poller.poll_once()
        
        self.assertTrue(result)
        mock_fetch_em.assert_called()
        mock_fetch_noaa.assert_called()
        mock_update_cm.assert_called_once()

    def test_validate_config(self):
        """Test configuration validation"""
        # Valid config should not raise exception
        try:
            self.poller._validate_config()
        except Exception as e:
            self.fail(f"Valid config raised exception: {e}")
        
        # Invalid config should raise exception
        invalid_config = {}
        invalid_poller = CarbonIntensityPoller(invalid_config)
        
        with self.assertRaises(ValueError):
            invalid_poller._validate_config()


if __name__ == '__main__':
    unittest.main()