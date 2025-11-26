#!/usr/bin/env python3
"""
End-to-end integration tests for Carbon-Kube
"""

import unittest
import subprocess
import time
import json
import yaml
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException


class TestCarbonKubeE2E(unittest.TestCase):
    """End-to-end integration tests"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        try:
            # Load kubeconfig
            config.load_kube_config()
            cls.k8s_client = client.ApiV1()
            cls.apps_client = client.AppsV1Api()
            cls.namespace = 'carbon-kube-test'
            
            # Create test namespace
            cls._create_test_namespace()
            
        except Exception as e:
            raise unittest.SkipTest(f"Kubernetes not available: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        try:
            cls._cleanup_test_namespace()
        except Exception:
            pass

    @classmethod
    def _create_test_namespace(cls):
        """Create test namespace"""
        namespace = client.V1Namespace(
            metadata=client.V1ObjectMeta(name=cls.namespace)
        )
        try:
            cls.k8s_client.create_namespace(namespace)
        except ApiException as e:
            if e.status != 409:  # Already exists
                raise

    @classmethod
    def _cleanup_test_namespace(cls):
        """Clean up test namespace"""
        try:
            cls.k8s_client.delete_namespace(cls.namespace)
        except ApiException:
            pass

    def test_helm_chart_deployment(self):
        """Test Helm chart deployment"""
        # Install Carbon-Kube using Helm
        cmd = [
            'helm', 'install', 'carbon-kube-test',
            './charts/carbon-kube',
            '--namespace', self.namespace,
            '--set', 'global.debug=true',
            '--set', 'electricityMaps.apiKey=test-key',
            '--set', 'noaa.apiKey=test-key',
            '--wait', '--timeout=300s'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Helm install failed: {result.stderr}")
        
        # Verify deployment
        self._verify_deployment_ready('carbon-kube-scheduler')

    def test_configmap_creation(self):
        """Test ConfigMap creation and content"""
        # Check if ConfigMap exists
        try:
            cm = self.k8s_client.read_namespaced_config_map(
                'carbon-kube-config', self.namespace
            )
            self.assertIsNotNone(cm.data)
            self.assertIn('threshold', cm.data)
            
        except ApiException as e:
            self.fail(f"ConfigMap not found: {e}")

    def test_scheduler_plugin_registration(self):
        """Test scheduler plugin registration"""
        # Check scheduler deployment
        try:
            deployment = self.apps_client.read_namespaced_deployment(
                'carbon-kube-scheduler', self.namespace
            )
            
            # Verify scheduler config is mounted
            containers = deployment.spec.template.spec.containers
            scheduler_container = next(
                (c for c in containers if c.name == 'scheduler'), None
            )
            self.assertIsNotNone(scheduler_container)
            
            # Check volume mounts
            volume_mounts = scheduler_container.volume_mounts or []
            config_mount = next(
                (vm for vm in volume_mounts if vm.name == 'scheduler-config'), None
            )
            self.assertIsNotNone(config_mount)
            
        except ApiException as e:
            self.fail(f"Scheduler deployment not found: {e}")

    def test_carbon_intensity_data_update(self):
        """Test carbon intensity data updates"""
        # Wait for poller to run
        time.sleep(60)
        
        try:
            cm = self.k8s_client.read_namespaced_config_map(
                'carbon-intensity-data', self.namespace
            )
            
            self.assertIsNotNone(cm.data)
            self.assertIn('carbon-intensity', cm.data)
            
            # Parse carbon intensity data
            carbon_data = json.loads(cm.data['carbon-intensity'])
            self.assertIsInstance(carbon_data, dict)
            
            # Verify data structure
            for zone, data in carbon_data.items():
                self.assertIn('carbon_intensity', data)
                self.assertIn('timestamp', data)
                self.assertIsInstance(data['carbon_intensity'], (int, float))
                
        except ApiException as e:
            self.fail(f"Carbon intensity ConfigMap not found: {e}")

    def test_pod_scheduling_with_carbon_awareness(self):
        """Test pod scheduling with carbon awareness"""
        # Create test pod with specific requirements
        test_pod = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name='carbon-test-pod',
                namespace=self.namespace,
                labels={'app': 'carbon-test'}
            ),
            spec=client.V1PodSpec(
                containers=[
                    client.V1Container(
                        name='test-container',
                        image='nginx:alpine',
                        resources=client.V1ResourceRequirements(
                            requests={'cpu': '100m', 'memory': '128Mi'}
                        )
                    )
                ],
                scheduler_name='carbon-kube-scheduler'
            )
        )
        
        # Create pod
        try:
            self.k8s_client.create_namespaced_pod(self.namespace, test_pod)
            
            # Wait for scheduling
            time.sleep(30)
            
            # Check pod status
            pod = self.k8s_client.read_namespaced_pod('carbon-test-pod', self.namespace)
            self.assertIsNotNone(pod.spec.node_name, "Pod was not scheduled")
            
            # Clean up
            self.k8s_client.delete_namespaced_pod('carbon-test-pod', self.namespace)
            
        except ApiException as e:
            self.fail(f"Pod scheduling test failed: {e}")

    def test_metrics_endpoint(self):
        """Test metrics endpoint availability"""
        # Port-forward to scheduler service
        port_forward_cmd = [
            'kubectl', 'port-forward',
            f'service/carbon-kube-scheduler',
            '8080:8080',
            '-n', self.namespace
        ]
        
        port_forward_proc = subprocess.Popen(
            port_forward_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        try:
            # Wait for port-forward to establish
            time.sleep(5)
            
            # Test metrics endpoint
            response = requests.get('http://localhost:8080/metrics', timeout=10)
            self.assertEqual(response.status_code, 200)
            
            # Verify carbon-kube metrics are present
            metrics_text = response.text
            self.assertIn('carbon_kube_', metrics_text)
            
        except requests.RequestException as e:
            self.fail(f"Metrics endpoint not accessible: {e}")
        finally:
            port_forward_proc.terminate()
            port_forward_proc.wait()

    def test_rl_tuner_cronjob(self):
        """Test RL tuner CronJob execution"""
        try:
            # Check if CronJob exists
            batch_client = client.BatchV1Api()
            cronjob = batch_client.read_namespaced_cron_job(
                'carbon-kube-rl-tuner', self.namespace
            )
            
            self.assertIsNotNone(cronjob)
            self.assertEqual(cronjob.spec.schedule, '0 */6 * * *')
            
            # Trigger manual job execution
            job_name = 'carbon-kube-rl-tuner-manual'
            job = client.V1Job(
                metadata=client.V1ObjectMeta(name=job_name),
                spec=cronjob.spec.job_template.spec
            )
            
            batch_client.create_namespaced_job(self.namespace, job)
            
            # Wait for job completion
            timeout = 300  # 5 minutes
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                job_status = batch_client.read_namespaced_job(job_name, self.namespace)
                if job_status.status.succeeded:
                    break
                elif job_status.status.failed:
                    self.fail("RL tuner job failed")
                time.sleep(10)
            else:
                self.fail("RL tuner job did not complete within timeout")
            
            # Clean up
            batch_client.delete_namespaced_job(job_name, self.namespace)
            
        except ApiException as e:
            self.fail(f"RL tuner CronJob test failed: {e}")

    def test_prometheus_integration(self):
        """Test Prometheus integration"""
        if not self._is_prometheus_available():
            self.skipTest("Prometheus not available")
        
        # Check ServiceMonitor
        try:
            # This would require prometheus-operator CRDs
            # For now, just verify the ServiceMonitor YAML is valid
            with open('./charts/carbon-kube/templates/servicemonitor.yaml', 'r') as f:
                servicemonitor = yaml.safe_load(f)
                self.assertEqual(servicemonitor['kind'], 'ServiceMonitor')
                
        except FileNotFoundError:
            self.fail("ServiceMonitor template not found")

    def test_grafana_dashboards(self):
        """Test Grafana dashboard configuration"""
        dashboard_files = [
            './charts/carbon-kube/dashboards/carbon-kube-overview.json',
            './charts/carbon-kube/dashboards/carbon-kube-rl-tuner.json',
            './charts/carbon-kube/dashboards/carbon-kube-scheduler.json'
        ]
        
        for dashboard_file in dashboard_files:
            try:
                with open(dashboard_file, 'r') as f:
                    dashboard = json.load(f)
                    self.assertIn('title', dashboard)
                    self.assertIn('panels', dashboard)
                    self.assertGreater(len(dashboard['panels']), 0)
                    
            except FileNotFoundError:
                self.fail(f"Dashboard file not found: {dashboard_file}")
            except json.JSONDecodeError:
                self.fail(f"Invalid JSON in dashboard: {dashboard_file}")

    def _verify_deployment_ready(self, deployment_name):
        """Verify deployment is ready"""
        timeout = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_client.read_namespaced_deployment(
                    deployment_name, self.namespace
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas >= deployment.spec.replicas):
                    return True
                    
            except ApiException:
                pass
            
            time.sleep(10)
        
        self.fail(f"Deployment {deployment_name} not ready within timeout")

    def _is_prometheus_available(self):
        """Check if Prometheus is available in cluster"""
        try:
            # Try to find Prometheus service
            services = self.k8s_client.list_service_for_all_namespaces()
            for service in services.items:
                if 'prometheus' in service.metadata.name.lower():
                    return True
            return False
        except Exception:
            return False


class TestCarbonKubePerformance(unittest.TestCase):
    """Performance tests for Carbon-Kube"""

    def setUp(self):
        """Set up performance test environment"""
        try:
            config.load_kube_config()
            self.k8s_client = client.ApiV1()
            self.namespace = 'carbon-kube-perf-test'
        except Exception as e:
            raise unittest.SkipTest(f"Kubernetes not available: {e}")

    def test_scheduler_latency(self):
        """Test scheduler latency under load"""
        # Create multiple pods simultaneously
        pod_count = 50
        pods = []
        
        start_time = time.time()
        
        for i in range(pod_count):
            pod = client.V1Pod(
                metadata=client.V1ObjectMeta(
                    name=f'perf-test-pod-{i}',
                    namespace=self.namespace
                ),
                spec=client.V1PodSpec(
                    containers=[
                        client.V1Container(
                            name='test-container',
                            image='nginx:alpine',
                            resources=client.V1ResourceRequirements(
                                requests={'cpu': '10m', 'memory': '32Mi'}
                            )
                        )
                    ],
                    scheduler_name='carbon-kube-scheduler'
                )
            )
            pods.append(pod)
        
        # Create all pods
        for pod in pods:
            try:
                self.k8s_client.create_namespaced_pod(self.namespace, pod)
            except ApiException:
                pass  # Continue with other pods
        
        # Wait for all pods to be scheduled
        scheduled_count = 0
        timeout = 300  # 5 minutes
        
        while time.time() - start_time < timeout and scheduled_count < pod_count:
            scheduled_count = 0
            try:
                pod_list = self.k8s_client.list_namespaced_pod(self.namespace)
                for pod in pod_list.items:
                    if pod.spec.node_name:
                        scheduled_count += 1
            except ApiException:
                pass
            
            time.sleep(5)
        
        total_time = time.time() - start_time
        avg_scheduling_time = total_time / scheduled_count if scheduled_count > 0 else float('inf')
        
        # Clean up
        try:
            self.k8s_client.delete_collection_namespaced_pod(self.namespace)
        except ApiException:
            pass
        
        # Assert performance requirements
        self.assertGreater(scheduled_count, pod_count * 0.8, "Less than 80% of pods were scheduled")
        self.assertLess(avg_scheduling_time, 10.0, "Average scheduling time too high")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)