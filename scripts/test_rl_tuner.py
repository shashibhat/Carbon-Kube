#!/usr/bin/env python3
"""
Unit tests for the RL Tuner
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os
import json
from datetime import datetime, timezone
from rl_tuner import RLTuner, MigrationEvent, RLState, CarbonMigrationEnv


class TestMigrationEvent(unittest.TestCase):
    """Test cases for MigrationEvent dataclass"""

    def test_migration_event_creation(self):
        """Test MigrationEvent creation and serialization"""
        event = MigrationEvent(
            timestamp=datetime.now(timezone.utc),
            source_zone='US-CA',
            target_zone='US-TX',
            carbon_intensity_source=300.0,
            carbon_intensity_target=200.0,
            pod_cpu_request=2.0,
            pod_memory_request=4.0,
            migration_success=True,
            migration_duration=30.5,
            threshold_used=250.0
        )
        
        self.assertEqual(event.source_zone, 'US-CA')
        self.assertEqual(event.target_zone, 'US-TX')
        self.assertTrue(event.migration_success)
        
        # Test serialization
        event_dict = event.to_dict()
        self.assertIn('timestamp', event_dict)
        self.assertEqual(event_dict['source_zone'], 'US-CA')


class TestRLState(unittest.TestCase):
    """Test cases for RLState dataclass"""

    def test_rl_state_creation(self):
        """Test RLState creation and array conversion"""
        state = RLState(
            current_carbon_intensity=250.0,
            avg_carbon_intensity_1h=275.0,
            avg_carbon_intensity_24h=300.0,
            pending_migrations=5,
            successful_migrations_1h=10,
            failed_migrations_1h=2,
            cluster_utilization=0.75,
            time_of_day=14.5,
            day_of_week=2
        )
        
        self.assertEqual(state.current_carbon_intensity, 250.0)
        self.assertEqual(state.pending_migrations, 5)
        
        # Test array conversion
        state_array = state.to_array()
        self.assertEqual(len(state_array), 9)
        self.assertEqual(state_array[0], 250.0)


class TestCarbonMigrationEnv(unittest.TestCase):
    """Test cases for CarbonMigrationEnv"""

    def setUp(self):
        """Set up test environment"""
        self.env = CarbonMigrationEnv()

    def test_env_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.action_space.n, 21)  # 0-1000 in steps of 50
        self.assertEqual(len(self.env.observation_space.high), 9)

    def test_reset(self):
        """Test environment reset"""
        obs = self.env.reset()
        self.assertEqual(len(obs), 9)
        self.assertIsInstance(obs, np.ndarray)

    def test_step(self):
        """Test environment step"""
        self.env.reset()
        action = 10  # threshold = 500
        
        obs, reward, done, info = self.env.step(action)
        
        self.assertEqual(len(obs), 9)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_action_to_threshold(self):
        """Test action to threshold conversion"""
        self.assertEqual(self.env._action_to_threshold(0), 0)
        self.assertEqual(self.env._action_to_threshold(10), 500)
        self.assertEqual(self.env._action_to_threshold(20), 1000)

    def test_calculate_reward(self):
        """Test reward calculation"""
        # Test positive reward (carbon reduction)
        reward = self.env._calculate_reward(300, 200, True, 30)
        self.assertGreater(reward, 0)
        
        # Test negative reward (failed migration)
        reward = self.env._calculate_reward(300, 200, False, 30)
        self.assertLess(reward, 0)


class TestRLTuner(unittest.TestCase):
    """Test cases for RLTuner"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'model': {
                'algorithm': 'PPO',
                'learning_rate': 0.0003,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            },
            'environment': {
                'buffer_size': 10000,
                'min_buffer_size': 1000,
                'update_frequency': 100
            },
            'output': {
                'configmap_name': 'carbon-kube-config',
                'configmap_namespace': 'carbon-kube',
                'threshold_key': 'threshold'
            },
            'kubernetes': {
                'namespace': 'carbon-kube'
            }
        }
        
        # Create temporary directory for model storage
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model')
        
        self.tuner = RLTuner(self.config, model_path=self.model_path)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test RLTuner initialization"""
        self.assertIsNotNone(self.tuner.env)
        self.assertIsNotNone(self.tuner.model)
        self.assertEqual(len(self.tuner.replay_buffer), 0)

    @patch('kubernetes.client.CoreV1Api')
    def test_collect_migration_events(self, mock_k8s_api):
        """Test migration event collection from Kubernetes"""
        mock_api_instance = Mock()
        mock_k8s_api.return_value = mock_api_instance
        
        # Mock events
        mock_event = Mock()
        mock_event.metadata.creation_timestamp = datetime.now(timezone.utc)
        mock_event.reason = 'CarbonMigration'
        mock_event.message = json.dumps({
            'source_zone': 'US-CA',
            'target_zone': 'US-TX',
            'carbon_intensity_source': 300.0,
            'carbon_intensity_target': 200.0,
            'success': True,
            'duration': 30.5,
            'threshold': 250.0,
            'pod_cpu': 2.0,
            'pod_memory': 4.0
        })
        
        mock_api_instance.list_namespaced_event.return_value.items = [mock_event]
        
        events = self.tuner._collect_migration_events()
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source_zone, 'US-CA')
        self.assertTrue(events[0].migration_success)

    def test_add_to_replay_buffer(self):
        """Test adding events to replay buffer"""
        event = MigrationEvent(
            timestamp=datetime.now(timezone.utc),
            source_zone='US-CA',
            target_zone='US-TX',
            carbon_intensity_source=300.0,
            carbon_intensity_target=200.0,
            pod_cpu_request=2.0,
            pod_memory_request=4.0,
            migration_success=True,
            migration_duration=30.5,
            threshold_used=250.0
        )
        
        initial_size = len(self.tuner.replay_buffer)
        self.tuner._add_to_replay_buffer(event)
        
        self.assertEqual(len(self.tuner.replay_buffer), initial_size + 1)

    def test_create_state_from_event(self):
        """Test state creation from migration event"""
        event = MigrationEvent(
            timestamp=datetime.now(timezone.utc),
            source_zone='US-CA',
            target_zone='US-TX',
            carbon_intensity_source=300.0,
            carbon_intensity_target=200.0,
            pod_cpu_request=2.0,
            pod_memory_request=4.0,
            migration_success=True,
            migration_duration=30.5,
            threshold_used=250.0
        )
        
        state = self.tuner._create_state_from_event(event)
        
        self.assertIsInstance(state, RLState)
        self.assertEqual(state.current_carbon_intensity, 300.0)

    def test_model_save_load(self):
        """Test model saving and loading"""
        # Save model
        self.tuner.save_model()
        self.assertTrue(os.path.exists(f"{self.model_path}.zip"))
        
        # Create new tuner and load model
        new_tuner = RLTuner(self.config, model_path=self.model_path)
        new_tuner.load_model()
        
        # Models should have same parameters
        self.assertEqual(
            self.tuner.model.learning_rate,
            new_tuner.model.learning_rate
        )

    @patch.object(RLTuner, '_collect_migration_events')
    def test_update_replay_buffer(self, mock_collect):
        """Test replay buffer update"""
        mock_event = MigrationEvent(
            timestamp=datetime.now(timezone.utc),
            source_zone='US-CA',
            target_zone='US-TX',
            carbon_intensity_source=300.0,
            carbon_intensity_target=200.0,
            pod_cpu_request=2.0,
            pod_memory_request=4.0,
            migration_success=True,
            migration_duration=30.5,
            threshold_used=250.0
        )
        mock_collect.return_value = [mock_event]
        
        initial_size = len(self.tuner.replay_buffer)
        self.tuner.update_replay_buffer()
        
        self.assertEqual(len(self.tuner.replay_buffer), initial_size + 1)

    def test_get_recommended_threshold(self):
        """Test threshold recommendation"""
        # Create mock state
        state = RLState(
            current_carbon_intensity=250.0,
            avg_carbon_intensity_1h=275.0,
            avg_carbon_intensity_24h=300.0,
            pending_migrations=5,
            successful_migrations_1h=10,
            failed_migrations_1h=2,
            cluster_utilization=0.75,
            time_of_day=14.5,
            day_of_week=2
        )
        
        threshold = self.tuner.get_recommended_threshold(state)
        
        self.assertIsInstance(threshold, float)
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 1000)

    @patch('kubernetes.client.CoreV1Api')
    def test_update_threshold_configmap(self, mock_k8s_api):
        """Test threshold update in ConfigMap"""
        mock_api_instance = Mock()
        mock_k8s_api.return_value = mock_api_instance
        
        # Mock existing ConfigMap
        existing_cm = Mock()
        existing_cm.data = {'threshold': '250'}
        mock_api_instance.read_namespaced_config_map.return_value = existing_cm
        
        result = self.tuner._update_threshold_configmap(300.0)
        
        self.assertTrue(result)
        mock_api_instance.patch_namespaced_config_map.assert_called_once()

    def test_validate_config(self):
        """Test configuration validation"""
        # Valid config should not raise exception
        try:
            self.tuner._validate_config()
        except Exception as e:
            self.fail(f"Valid config raised exception: {e}")
        
        # Invalid config should raise exception
        invalid_config = {}
        
        with self.assertRaises(ValueError):
            RLTuner(invalid_config)


if __name__ == '__main__':
    unittest.main()