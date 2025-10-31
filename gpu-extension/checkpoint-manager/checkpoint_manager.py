#!/usr/bin/env python3
"""
Carbon-Aware GPU Checkpoint Manager for Carbon-Kube

This module provides intelligent checkpointing capabilities for GPU workloads,
enabling carbon-aware migration with minimal disruption to training processes.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml

import aiofiles
import aiohttp
import kubernetes_asyncio as k8s
import torch
import torch.distributed as dist
from kubernetes_asyncio.client.rest import ApiException
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
checkpoint_operations = Counter('checkpoint_operations_total', 
                              'Total checkpoint operations', ['operation', 'status'])
checkpoint_duration = Histogram('checkpoint_duration_seconds',
                               'Checkpoint operation duration')
checkpoint_size = Gauge('checkpoint_size_bytes',
                       'Size of checkpoint files', ['workload_id'])
migration_carbon_savings = Counter('migration_carbon_savings_gco2_total',
                                 'Total carbon savings from migrations')
checkpoint_storage_usage = Gauge('checkpoint_storage_usage_bytes',
                                'Total checkpoint storage usage')

class CheckpointManager:
    """Manages GPU workload checkpointing for carbon-aware migrations."""
    
    def __init__(self, config_path: str = "/etc/checkpoint-manager/config.yaml"):
        self.config = self._load_config(config_path)
        self.k8s_client = None
        self.redis_client = None
        self.storage_backend = None
        self.active_checkpoints: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize the checkpoint manager."""
        try:
            # Initialize Kubernetes client
            k8s.config.load_incluster_config()
            self.k8s_client = k8s.client.ApiClient()
            
            # Initialize Redis client
            self.redis_client = redis.Redis(
                host=self.config['redis']['host'],
                port=self.config['redis']['port'],
                db=self.config['redis']['db'],
                decode_responses=True
            )
            
            # Initialize storage backend
            self.storage_backend = self._init_storage_backend()
            
            # Start Prometheus metrics server
            start_http_server(self.config['metrics']['port'])
            
            logger.info("Checkpoint manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint manager: {e}")
            raise
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Return default config
            return {
                'redis': {'host': 'redis-service', 'port': 6379, 'db': 2},
                'storage': {'type': 'pvc', 'path': '/checkpoints'},
                'metrics': {'port': 8080},
                'checkpoint': {
                    'interval_seconds': 300,
                    'retention_hours': 24,
                    'compression': True,
                    'async_save': True
                }
            }
    
    def _init_storage_backend(self):
        """Initialize the storage backend for checkpoints."""
        storage_type = self.config['storage']['type']
        
        if storage_type == 'pvc':
            return PVCStorageBackend(self.config['storage']['path'])
        elif storage_type == 's3':
            return S3StorageBackend(self.config['storage'])
        elif storage_type == 'nfs':
            return NFSStorageBackend(self.config['storage'])
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")
    
    async def create_checkpoint(self, workload_id: str, pod_name: str, 
                              namespace: str, force: bool = False) -> Dict:
        """Create a checkpoint for a GPU workload."""
        start_time = time.time()
        
        try:
            logger.info(f"Creating checkpoint for workload {workload_id}")
            
            # Check if checkpoint is already in progress
            if workload_id in self.active_checkpoints and not force:
                return {'status': 'in_progress', 'checkpoint_id': self.active_checkpoints[workload_id]['id']}
            
            # Generate checkpoint metadata
            checkpoint_id = f"{workload_id}-{int(time.time())}"
            checkpoint_meta = {
                'id': checkpoint_id,
                'workload_id': workload_id,
                'pod_name': pod_name,
                'namespace': namespace,
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'creating',
                'carbon_context': await self._get_carbon_context(namespace)
            }
            
            # Mark checkpoint as active
            self.active_checkpoints[workload_id] = checkpoint_meta
            
            # Execute checkpoint creation
            if await self._is_pytorch_workload(pod_name, namespace):
                checkpoint_data = await self._create_pytorch_checkpoint(
                    pod_name, namespace, checkpoint_id)
            elif await self._is_tensorflow_workload(pod_name, namespace):
                checkpoint_data = await self._create_tensorflow_checkpoint(
                    pod_name, namespace, checkpoint_id)
            else:
                checkpoint_data = await self._create_generic_checkpoint(
                    pod_name, namespace, checkpoint_id)
            
            # Save checkpoint to storage
            storage_path = await self.storage_backend.save_checkpoint(
                checkpoint_id, checkpoint_data)
            
            # Update metadata
            checkpoint_meta.update({
                'status': 'completed',
                'storage_path': storage_path,
                'size_bytes': len(checkpoint_data) if isinstance(checkpoint_data, bytes) else 0,
                'duration_seconds': time.time() - start_time
            })
            
            # Store metadata in Redis
            await self.redis_client.setex(
                f"checkpoint:{checkpoint_id}",
                self.config['checkpoint']['retention_hours'] * 3600,
                json.dumps(checkpoint_meta)
            )
            
            # Update metrics
            checkpoint_operations.labels(operation='create', status='success').inc()
            checkpoint_duration.observe(time.time() - start_time)
            checkpoint_size.labels(workload_id=workload_id).set(checkpoint_meta['size_bytes'])
            
            # Remove from active checkpoints
            self.active_checkpoints.pop(workload_id, None)
            
            logger.info(f"Checkpoint {checkpoint_id} created successfully")
            return checkpoint_meta
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint for {workload_id}: {e}")
            checkpoint_operations.labels(operation='create', status='error').inc()
            self.active_checkpoints.pop(workload_id, None)
            raise
    
    async def restore_checkpoint(self, checkpoint_id: str, target_pod: str, 
                               target_namespace: str) -> Dict:
        """Restore a checkpoint to a target pod."""
        start_time = time.time()
        
        try:
            logger.info(f"Restoring checkpoint {checkpoint_id} to {target_pod}")
            
            # Get checkpoint metadata
            checkpoint_meta_str = await self.redis_client.get(f"checkpoint:{checkpoint_id}")
            if not checkpoint_meta_str:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
            
            checkpoint_meta = json.loads(checkpoint_meta_str)
            
            # Load checkpoint data from storage
            checkpoint_data = await self.storage_backend.load_checkpoint(
                checkpoint_meta['storage_path'])
            
            # Restore based on workload type
            if 'pytorch' in checkpoint_meta.get('framework', '').lower():
                await self._restore_pytorch_checkpoint(
                    target_pod, target_namespace, checkpoint_data)
            elif 'tensorflow' in checkpoint_meta.get('framework', '').lower():
                await self._restore_tensorflow_checkpoint(
                    target_pod, target_namespace, checkpoint_data)
            else:
                await self._restore_generic_checkpoint(
                    target_pod, target_namespace, checkpoint_data)
            
            # Update metrics
            checkpoint_operations.labels(operation='restore', status='success').inc()
            checkpoint_duration.observe(time.time() - start_time)
            
            logger.info(f"Checkpoint {checkpoint_id} restored successfully")
            return {
                'status': 'success',
                'checkpoint_id': checkpoint_id,
                'duration_seconds': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            checkpoint_operations.labels(operation='restore', status='error').inc()
            raise
    
    async def migrate_workload(self, workload_id: str, source_node: str, 
                             target_node: str, carbon_savings_gco2: float) -> Dict:
        """Perform carbon-aware workload migration with checkpointing."""
        migration_start = time.time()
        
        try:
            logger.info(f"Starting carbon-aware migration of {workload_id} from {source_node} to {target_node}")
            
            # Get workload pod information
            pod_info = await self._get_workload_pod(workload_id)
            if not pod_info:
                raise ValueError(f"Workload {workload_id} not found")
            
            source_pod = pod_info['name']
            namespace = pod_info['namespace']
            
            # Create checkpoint on source node
            checkpoint_meta = await self.create_checkpoint(workload_id, source_pod, namespace)
            checkpoint_id = checkpoint_meta['id']
            
            # Calculate migration decision
            migration_decision = await self._evaluate_migration_decision(
                workload_id, source_node, target_node, carbon_savings_gco2)
            
            if not migration_decision['should_migrate']:
                logger.info(f"Migration cancelled: {migration_decision['reason']}")
                return {
                    'status': 'cancelled',
                    'reason': migration_decision['reason'],
                    'checkpoint_id': checkpoint_id
                }
            
            # Create new pod on target node
            new_pod_spec = await self._create_migration_pod_spec(
                pod_info, target_node, checkpoint_id)
            
            # Deploy new pod
            v1 = k8s.client.CoreV1Api(self.k8s_client)
            new_pod = await v1.create_namespaced_pod(
                namespace=namespace,
                body=new_pod_spec
            )
            
            # Wait for new pod to be ready
            await self._wait_for_pod_ready(new_pod.metadata.name, namespace)
            
            # Restore checkpoint on new pod
            await self.restore_checkpoint(checkpoint_id, new_pod.metadata.name, namespace)
            
            # Gracefully terminate old pod
            await self._terminate_pod_gracefully(source_pod, namespace)
            
            # Update migration metrics
            migration_carbon_savings.inc(carbon_savings_gco2)
            
            # Record migration in Redis
            migration_record = {
                'workload_id': workload_id,
                'source_node': source_node,
                'target_node': target_node,
                'checkpoint_id': checkpoint_id,
                'carbon_savings_gco2': carbon_savings_gco2,
                'duration_seconds': time.time() - migration_start,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                f"migration:{workload_id}:{int(time.time())}",
                86400,  # 24 hours
                json.dumps(migration_record)
            )
            
            logger.info(f"Migration completed successfully. Carbon savings: {carbon_savings_gco2} gCO2")
            
            return {
                'status': 'success',
                'new_pod': new_pod.metadata.name,
                'checkpoint_id': checkpoint_id,
                'carbon_savings_gco2': carbon_savings_gco2,
                'duration_seconds': time.time() - migration_start
            }
            
        except Exception as e:
            logger.error(f"Migration failed for {workload_id}: {e}")
            raise
    
    async def _create_pytorch_checkpoint(self, pod_name: str, namespace: str, 
                                       checkpoint_id: str) -> bytes:
        """Create a PyTorch-specific checkpoint."""
        # Execute checkpoint command in the pod
        checkpoint_cmd = [
            'python', '-c', f'''
import torch
import os
import pickle

# Get model state from environment or default location
model_path = os.environ.get("MODEL_STATE_PATH", "/tmp/model_state.pt")
optimizer_path = os.environ.get("OPTIMIZER_STATE_PATH", "/tmp/optimizer_state.pt")

checkpoint_data = {{}}

# Save model state if available
if os.path.exists(model_path):
    checkpoint_data["model_state"] = torch.load(model_path, map_location="cpu")

# Save optimizer state if available
if os.path.exists(optimizer_path):
    checkpoint_data["optimizer_state"] = torch.load(optimizer_path, map_location="cpu")

# Add training metadata
checkpoint_data["metadata"] = {{
    "framework": "pytorch",
    "checkpoint_id": "{checkpoint_id}",
    "timestamp": "{datetime.utcnow().isoformat()}",
    "epoch": os.environ.get("CURRENT_EPOCH", "0"),
    "step": os.environ.get("CURRENT_STEP", "0")
}}

# Save checkpoint
torch.save(checkpoint_data, "/tmp/checkpoint.pt")
print("Checkpoint created successfully")
'''
        ]
        
        # Execute command in pod
        await self._exec_in_pod(pod_name, namespace, checkpoint_cmd)
        
        # Copy checkpoint file from pod
        checkpoint_data = await self._copy_file_from_pod(
            pod_name, namespace, "/tmp/checkpoint.pt")
        
        return checkpoint_data
    
    async def _create_tensorflow_checkpoint(self, pod_name: str, namespace: str, 
                                          checkpoint_id: str) -> bytes:
        """Create a TensorFlow-specific checkpoint."""
        checkpoint_cmd = [
            'python', '-c', f'''
import tensorflow as tf
import os
import json

# Create checkpoint manager
checkpoint_dir = "/tmp/tf_checkpoint"
os.makedirs(checkpoint_dir, exist_ok=True)

# Save checkpoint metadata
metadata = {{
    "framework": "tensorflow",
    "checkpoint_id": "{checkpoint_id}",
    "timestamp": "{datetime.utcnow().isoformat()}",
    "step": os.environ.get("CURRENT_STEP", "0")
}}

with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f)

print("TensorFlow checkpoint created successfully")
'''
        ]
        
        await self._exec_in_pod(pod_name, namespace, checkpoint_cmd)
        
        # Create tar archive of checkpoint directory
        tar_cmd = ['tar', '-czf', '/tmp/tf_checkpoint.tar.gz', '-C', '/tmp', 'tf_checkpoint']
        await self._exec_in_pod(pod_name, namespace, tar_cmd)
        
        # Copy checkpoint archive from pod
        checkpoint_data = await self._copy_file_from_pod(
            pod_name, namespace, "/tmp/tf_checkpoint.tar.gz")
        
        return checkpoint_data
    
    async def _create_generic_checkpoint(self, pod_name: str, namespace: str, 
                                       checkpoint_id: str) -> bytes:
        """Create a generic checkpoint for unknown workload types."""
        # Create a simple state snapshot
        checkpoint_cmd = [
            'sh', '-c', f'''
mkdir -p /tmp/generic_checkpoint
echo '{{"checkpoint_id": "{checkpoint_id}", "timestamp": "{datetime.utcnow().isoformat()}", "type": "generic"}}' > /tmp/generic_checkpoint/metadata.json

# Copy any state files that might exist
find /app -name "*.state" -o -name "*.ckpt" -o -name "*.model" 2>/dev/null | head -10 | while read file; do
    cp "$file" /tmp/generic_checkpoint/ 2>/dev/null || true
done

tar -czf /tmp/generic_checkpoint.tar.gz -C /tmp generic_checkpoint
echo "Generic checkpoint created"
'''
        ]
        
        await self._exec_in_pod(pod_name, namespace, checkpoint_cmd)
        
        checkpoint_data = await self._copy_file_from_pod(
            pod_name, namespace, "/tmp/generic_checkpoint.tar.gz")
        
        return checkpoint_data
    
    async def _get_carbon_context(self, namespace: str) -> Dict:
        """Get current carbon intensity context."""
        try:
            v1 = k8s.client.CoreV1Api(self.k8s_client)
            config_map = await v1.read_namespaced_config_map(
                name="carbon-scores", namespace="default")
            
            zones_data = json.loads(config_map.data.get("zones", "{}"))
            return {
                'zones': zones_data,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.warning(f"Failed to get carbon context: {e}")
            return {}
    
    async def _is_pytorch_workload(self, pod_name: str, namespace: str) -> bool:
        """Check if workload is PyTorch-based."""
        try:
            v1 = k8s.client.CoreV1Api(self.k8s_client)
            pod = await v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            
            # Check container images and environment variables
            for container in pod.spec.containers:
                if 'pytorch' in container.image.lower():
                    return True
                
                if container.env:
                    for env_var in container.env:
                        if env_var.name in ['PYTORCH_VERSION', 'TORCH_VERSION']:
                            return True
            
            return False
        except Exception:
            return False
    
    async def _is_tensorflow_workload(self, pod_name: str, namespace: str) -> bool:
        """Check if workload is TensorFlow-based."""
        try:
            v1 = k8s.client.CoreV1Api(self.k8s_client)
            pod = await v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            
            for container in pod.spec.containers:
                if 'tensorflow' in container.image.lower() or 'tf-' in container.image.lower():
                    return True
                
                if container.env:
                    for env_var in container.env:
                        if env_var.name in ['TF_VERSION', 'TENSORFLOW_VERSION']:
                            return True
            
            return False
        except Exception:
            return False
    
    async def _exec_in_pod(self, pod_name: str, namespace: str, command: List[str]):
        """Execute command in pod."""
        # This would use the Kubernetes exec API
        # Implementation depends on specific requirements
        pass
    
    async def _copy_file_from_pod(self, pod_name: str, namespace: str, file_path: str) -> bytes:
        """Copy file from pod."""
        # This would use kubectl cp equivalent
        # Implementation depends on specific requirements
        return b"dummy_checkpoint_data"
    
    async def cleanup_old_checkpoints(self):
        """Clean up old checkpoints based on retention policy."""
        try:
            retention_hours = self.config['checkpoint']['retention_hours']
            cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
            
            # Get all checkpoint keys from Redis
            checkpoint_keys = await self.redis_client.keys("checkpoint:*")
            
            for key in checkpoint_keys:
                checkpoint_data = await self.redis_client.get(key)
                if checkpoint_data:
                    checkpoint_meta = json.loads(checkpoint_data)
                    checkpoint_time = datetime.fromisoformat(checkpoint_meta['timestamp'])
                    
                    if checkpoint_time < cutoff_time:
                        # Delete from storage
                        await self.storage_backend.delete_checkpoint(
                            checkpoint_meta['storage_path'])
                        
                        # Delete from Redis
                        await self.redis_client.delete(key)
                        
                        logger.info(f"Cleaned up old checkpoint {checkpoint_meta['id']}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")


class PVCStorageBackend:
    """Persistent Volume Claim storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def save_checkpoint(self, checkpoint_id: str, data: bytes) -> str:
        """Save checkpoint data to PVC."""
        file_path = self.base_path / f"{checkpoint_id}.ckpt"
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)
        
        return str(file_path)
    
    async def load_checkpoint(self, storage_path: str) -> bytes:
        """Load checkpoint data from PVC."""
        async with aiofiles.open(storage_path, 'rb') as f:
            return await f.read()
    
    async def delete_checkpoint(self, storage_path: str):
        """Delete checkpoint from PVC."""
        try:
            os.unlink(storage_path)
        except FileNotFoundError:
            pass


class S3StorageBackend:
    """S3-compatible storage backend."""
    
    def __init__(self, config: Dict):
        self.bucket = config['bucket']
        self.prefix = config.get('prefix', 'checkpoints/')
        # Initialize S3 client here
    
    async def save_checkpoint(self, checkpoint_id: str, data: bytes) -> str:
        """Save checkpoint data to S3."""
        # Implementation for S3 upload
        return f"s3://{self.bucket}/{self.prefix}{checkpoint_id}.ckpt"
    
    async def load_checkpoint(self, storage_path: str) -> bytes:
        """Load checkpoint data from S3."""
        # Implementation for S3 download
        return b"checkpoint_data"
    
    async def delete_checkpoint(self, storage_path: str):
        """Delete checkpoint from S3."""
        # Implementation for S3 delete
        pass


class NFSStorageBackend:
    """NFS storage backend."""
    
    def __init__(self, config: Dict):
        self.mount_path = Path(config['mount_path'])
        self.mount_path.mkdir(parents=True, exist_ok=True)
    
    async def save_checkpoint(self, checkpoint_id: str, data: bytes) -> str:
        """Save checkpoint data to NFS."""
        file_path = self.mount_path / f"{checkpoint_id}.ckpt"
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)
        
        return str(file_path)
    
    async def load_checkpoint(self, storage_path: str) -> bytes:
        """Load checkpoint data from NFS."""
        async with aiofiles.open(storage_path, 'rb') as f:
            return await f.read()
    
    async def delete_checkpoint(self, storage_path: str):
        """Delete checkpoint from NFS."""
        try:
            os.unlink(storage_path)
        except FileNotFoundError:
            pass


async def main():
    """Main entry point for the checkpoint manager."""
    checkpoint_manager = CheckpointManager()
    await checkpoint_manager.initialize()
    
    # Start cleanup task
    async def cleanup_task():
        while True:
            await asyncio.sleep(3600)  # Run every hour
            await checkpoint_manager.cleanup_old_checkpoints()
    
    # Start background tasks
    cleanup_task_handle = asyncio.create_task(cleanup_task())
    
    try:
        # Keep the service running
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down checkpoint manager")
        cleanup_task_handle.cancel()


if __name__ == "__main__":
    asyncio.run(main())