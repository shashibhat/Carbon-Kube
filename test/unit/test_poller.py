#!/usr/bin/env python3
"""
Unit tests for Carbon Intensity Poller
Tests the carbon intensity data fetching and ConfigMap updating functionality
"""

import json
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import sys
import os

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts'))

from poller import CarbonIntensityPoller, ZoneMapping, NOAAStation


class TestCarbonIntensityPoller:
    """Test suite for CarbonIntensityPoller class"""
    
    @pytest.fixture
    def poller(self):
        """Create a test poller instance"""
        return CarbonIntensityPoller()
    
    @pytest.fixture
    def mock_k8s_client(self):
        """Mock Kubernetes client"""
        mock_client = Mock()
        mock_v1 = Mock()
        mock_client.CoreV1Api.return_value = mock_v1
        return mock_client, mock_v1
    
    def test_init(self, poller):
        """Test poller initialization"""
        assert poller.electricity_maps_token is not None
        assert poller.noaa_token is not None
        assert poller.configmap_name == "carbon-intensity-data"
        assert poller.configmap_namespace == "carbon-kube"
        assert poller.default_intensity == 300.0
        assert poller.cache_duration == 300  # 5 minutes
    
    @patch('requests.get')
    def test_fetch_electricity_maps_success(self, mock_get, poller):
        """Test successful Electricity Maps API call"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'carbonIntensity': 250.5,
            'datetime': '2024-01-15T10:00:00.000Z',
            'forecast': [
                {'carbonIntensity': 240.0, 'datetime': '2024-01-15T11:00:00.000Z'},
                {'carbonIntensity': 260.0, 'datetime': '2024-01-15T12:00:00.000Z'},
            ]
        }
        mock_get.return_value = mock_response
        
        result = poller.fetch_electricity_maps_data('US-CA')
        
        assert result is not None
        assert result['intensity'] == 250.5
        assert result['zone'] == 'US-CA'
        assert len(result['forecast']) == 2
        assert result['forecast'][0] == 240.0
        
        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert 'zones/US-CA' in call_args[0][0]
    
    @patch('requests.get')
    def test_fetch_electricity_maps_failure(self, mock_get, poller):
        """Test Electricity Maps API failure"""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_get.return_value = mock_response
        
        result = poller.fetch_electricity_maps_data('INVALID-ZONE')
        
        assert result is None
    
    @patch('requests.get')
    def test_fetch_noaa_data_success(self, mock_get, poller):
        """Test successful NOAA API call"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': [
                {
                    'date': '2024-01-15T10:00:00',
                    'value': '15.5'  # Temperature in Celsius
                }
            ]
        }
        mock_get.return_value = mock_response
        
        result = poller.fetch_noaa_fallback_data('KORD')  # Chicago O'Hare
        
        assert result is not None
        assert result['intensity'] > 0  # Should calculate intensity from temperature
        assert result['zone'] == 'KORD'
        assert result['source'] == 'noaa_fallback'
    
    @patch('requests.get')
    def test_fetch_aws_carbon_data_success(self, mock_get, poller):
        """Test successful AWS Carbon Footprint API call"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'CarbonFootprintSummary': {
                'TimeUnit': 'MONTHLY',
                'CarbonFootprintByService': [
                    {
                        'ServiceName': 'AmazonEC2-Instance',
                        'CarbonFootprintByRegion': [
                            {
                                'Region': 'us-west-2',
                                'CarbonFootprint': {
                                    'Amount': 125.5,
                                    'Unit': 'MtCO2e'
                                }
                            }
                        ]
                    }
                ]
            }
        }
        mock_get.return_value = mock_response
        
        result = poller.fetch_aws_carbon_data('us-west-2')
        
        assert result is not None
        assert result['intensity'] > 0
        assert result['zone'] == 'us-west-2'
        assert result['source'] == 'aws_carbon_footprint'
    
    def test_calculate_intensity_from_temperature(self, poller):
        """Test carbon intensity calculation from temperature"""
        # Test various temperatures
        test_cases = [
            (20.0, 250.0),  # Moderate temperature
            (35.0, 400.0),  # High temperature (more cooling needed)
            (5.0, 350.0),   # Low temperature (more heating needed)
            (22.0, 260.0),  # Optimal temperature
        ]
        
        for temp, expected_min in test_cases:
            intensity = poller.calculate_intensity_from_temperature(temp)
            assert intensity >= expected_min
            assert intensity <= 500.0  # Max intensity cap
    
    def test_get_zone_for_region(self, poller):
        """Test zone mapping for AWS regions"""
        test_cases = [
            ('us-west-2', 'US-CA'),
            ('us-east-1', 'US-VA'),
            ('eu-west-1', 'IE'),
            ('ap-southeast-1', 'SG'),
            ('unknown-region', 'US')  # Default fallback
        ]
        
        for region, expected_zone in test_cases:
            zone = poller.get_zone_for_region(region)
            assert zone == expected_zone
    
    def test_get_noaa_station_for_region(self, poller):
        """Test NOAA station mapping for AWS regions"""
        test_cases = [
            ('us-west-2', 'KSEA'),  # Seattle
            ('us-east-1', 'KIAD'),  # Washington DC
            ('us-central-1', 'KORD'),  # Chicago (fallback)
        ]
        
        for region, expected_station in test_cases:
            station = poller.get_noaa_station_for_region(region)
            assert station == expected_station
    
    @patch('kubernetes.client.CoreV1Api')
    @patch('kubernetes.config.load_incluster_config')
    def test_update_configmap_success(self, mock_load_config, mock_core_v1, poller):
        """Test successful ConfigMap update"""
        # Mock Kubernetes client
        mock_api = Mock()
        mock_core_v1.return_value = mock_api
        
        # Mock existing ConfigMap
        mock_configmap = Mock()
        mock_configmap.data = {}
        mock_api.read_namespaced_config_map.return_value = mock_configmap
        
        # Test data
        intensity_data = {
            'us-west-2a': {
                'zone': 'us-west-2a',
                'intensity': 150.0,
                'timestamp': 1642248000,
                'source': 'electricity_maps',
                'forecast': [140.0, 145.0, 150.0]
            }
        }
        
        success = poller.update_configmap(intensity_data)
        
        assert success is True
        mock_api.patch_namespaced_config_map.assert_called_once()
        
        # Verify the data was properly serialized
        call_args = mock_api.patch_namespaced_config_map.call_args
        body = call_args[1]['body']
        assert 'zones' in body.data
        
        # Verify JSON serialization
        zones_data = json.loads(body.data['zones'])
        assert 'us-west-2a' in zones_data
        assert zones_data['us-west-2a']['intensity'] == 150.0
    
    @patch('kubernetes.client.CoreV1Api')
    @patch('kubernetes.config.load_incluster_config')
    def test_update_configmap_create_new(self, mock_load_config, mock_core_v1, poller):
        """Test ConfigMap creation when it doesn't exist"""
        # Mock Kubernetes client
        mock_api = Mock()
        mock_core_v1.return_value = mock_api
        
        # Mock ConfigMap not found
        from kubernetes.client.rest import ApiException
        mock_api.read_namespaced_config_map.side_effect = ApiException(status=404)
        
        intensity_data = {'test': {'intensity': 200.0}}
        success = poller.update_configmap(intensity_data)
        
        assert success is True
        mock_api.create_namespaced_config_map.assert_called_once()
    
    @patch('requests.get')
    def test_fetch_all_zones_data(self, mock_get, poller):
        """Test fetching data for all configured zones"""
        # Mock successful responses for different zones
        def mock_response_side_effect(url, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            
            if 'US-CA' in url:
                mock_response.json.return_value = {
                    'carbonIntensity': 150.0,
                    'datetime': '2024-01-15T10:00:00.000Z',
                    'forecast': []
                }
            elif 'US-TX' in url:
                mock_response.json.return_value = {
                    'carbonIntensity': 400.0,
                    'datetime': '2024-01-15T10:00:00.000Z',
                    'forecast': []
                }
            else:
                mock_response.status_code = 404
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
            
            return mock_response
        
        mock_get.side_effect = mock_response_side_effect
        
        # Test with subset of zones
        test_zones = ['US-CA', 'US-TX', 'INVALID']
        results = {}
        
        for zone in test_zones:
            data = poller.fetch_electricity_maps_data(zone)
            if data:
                results[zone] = data
        
        assert len(results) == 2  # Only valid zones
        assert 'US-CA' in results
        assert 'US-TX' in results
        assert results['US-CA']['intensity'] == 150.0
        assert results['US-TX']['intensity'] == 400.0
    
    def test_is_data_stale(self, poller):
        """Test data staleness detection"""
        current_time = datetime.now(timezone.utc).timestamp()
        
        # Fresh data (2 minutes old)
        fresh_timestamp = current_time - 120
        assert not poller.is_data_stale(fresh_timestamp)
        
        # Stale data (10 minutes old)
        stale_timestamp = current_time - 600
        assert poller.is_data_stale(stale_timestamp)
        
        # Edge case (exactly at cache duration)
        edge_timestamp = current_time - poller.cache_duration
        assert poller.is_data_stale(edge_timestamp)
    
    @patch('time.sleep')
    def test_retry_mechanism(self, mock_sleep, poller):
        """Test retry mechanism for failed API calls"""
        with patch('requests.get') as mock_get:
            # Mock consecutive failures then success
            mock_responses = []
            
            # First two calls fail
            for _ in range(2):
                mock_response = Mock()
                mock_response.status_code = 500
                mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
                mock_responses.append(mock_response)
            
            # Third call succeeds
            mock_success_response = Mock()
            mock_success_response.status_code = 200
            mock_success_response.json.return_value = {
                'carbonIntensity': 200.0,
                'datetime': '2024-01-15T10:00:00.000Z',
                'forecast': []
            }
            mock_responses.append(mock_success_response)
            
            mock_get.side_effect = mock_responses
            
            result = poller.fetch_electricity_maps_data('US-CA')
            
            # Should succeed after retries
            assert result is not None
            assert result['intensity'] == 200.0
            
            # Verify retry attempts
            assert mock_get.call_count == 3
            assert mock_sleep.call_count == 2  # Sleep between retries
    
    def test_data_validation(self, poller):
        """Test data validation and sanitization"""
        # Test valid data
        valid_data = {
            'intensity': 250.0,
            'zone': 'US-CA',
            'timestamp': 1642248000,
            'forecast': [240.0, 250.0, 260.0]
        }
        
        assert poller.validate_intensity_data(valid_data) is True
        
        # Test invalid data
        invalid_cases = [
            {'intensity': -50.0},  # Negative intensity
            {'intensity': 1500.0},  # Too high intensity
            {'zone': ''},  # Empty zone
            {'timestamp': 'invalid'},  # Invalid timestamp
            {}  # Empty data
        ]
        
        for invalid_data in invalid_cases:
            assert poller.validate_intensity_data(invalid_data) is False
    
    @patch('kubernetes.client.CoreV1Api')
    @patch('kubernetes.config.load_incluster_config')
    @patch('requests.get')
    def test_main_polling_loop(self, mock_get, mock_load_config, mock_core_v1, poller):
        """Test the main polling loop integration"""
        # Mock Kubernetes client
        mock_api = Mock()
        mock_core_v1.return_value = mock_api
        mock_configmap = Mock()
        mock_configmap.data = {}
        mock_api.read_namespaced_config_map.return_value = mock_configmap
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'carbonIntensity': 180.0,
            'datetime': '2024-01-15T10:00:00.000Z',
            'forecast': [175.0, 180.0, 185.0]
        }
        mock_get.return_value = mock_response
        
        # Run one iteration of polling
        success = poller.poll_and_update()
        
        assert success is True
        
        # Verify ConfigMap was updated
        mock_api.patch_namespaced_config_map.assert_called()


class TestZoneMapping:
    """Test zone mapping functionality"""
    
    def test_zone_mapping_completeness(self):
        """Test that all major AWS regions have zone mappings"""
        major_regions = [
            'us-east-1', 'us-west-1', 'us-west-2',
            'eu-west-1', 'eu-central-1',
            'ap-southeast-1', 'ap-northeast-1'
        ]
        
        for region in major_regions:
            # Should have a mapping (not default 'US')
            zone = None
            for mapping in ZoneMapping:
                if mapping['aws_region'] == region:
                    zone = mapping['electricity_maps_zone']
                    break
            
            assert zone is not None, f"No zone mapping for region {region}"
            assert zone != 'US', f"Region {region} uses default fallback"


class TestNOAAStation:
    """Test NOAA station functionality"""
    
    def test_noaa_station_completeness(self):
        """Test that all major AWS regions have NOAA station mappings"""
        major_regions = [
            'us-east-1', 'us-west-1', 'us-west-2'
        ]
        
        for region in major_regions:
            station = None
            for noaa_station in NOAAStation:
                if noaa_station['aws_region'] == region:
                    station = noaa_station['station_id']
                    break
            
            assert station is not None, f"No NOAA station for region {region}"


# Integration test fixtures
@pytest.fixture
def integration_poller():
    """Create poller for integration tests"""
    return CarbonIntensityPoller()


class TestIntegration:
    """Integration tests (require actual API access)"""
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        os.getenv('ELECTRICITY_MAPS_TOKEN') is None,
        reason="Requires ELECTRICITY_MAPS_TOKEN environment variable"
    )
    def test_real_electricity_maps_api(self, integration_poller):
        """Test with real Electricity Maps API (requires token)"""
        result = integration_poller.fetch_electricity_maps_data('US-CA')
        
        if result:  # API might be down or rate limited
            assert result['intensity'] > 0
            assert result['zone'] == 'US-CA'
            assert 'timestamp' in result
    
    @pytest.mark.integration
    @pytest.mark.skipif(
        os.getenv('NOAA_TOKEN') is None,
        reason="Requires NOAA_TOKEN environment variable"
    )
    def test_real_noaa_api(self, integration_poller):
        """Test with real NOAA API (requires token)"""
        result = integration_poller.fetch_noaa_fallback_data('KSEA')
        
        if result:  # API might be down or rate limited
            assert result['intensity'] > 0
            assert result['zone'] == 'KSEA'
            assert result['source'] == 'noaa_fallback'


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])