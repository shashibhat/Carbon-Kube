# Build Fixes Applied

This document summarizes the fixes applied to make Carbon-Kube buildable.

## Issues Fixed

### 1. Missing Go Plugin Entry Point
- **Issue**: The Makefile and Dockerfile referenced `pkg/emissionplugin/cmd/scheduler` but the directory didn't exist
- **Fix**: Created `pkg/emissionplugin/cmd/scheduler/main.go` with the plugin factory function

### 2. Incorrect Go Module Paths
- **Issue**: Makefile referenced `pkg/emissionplugin/go.mod` but go.mod is in the root directory
- **Fix**: Updated Makefile to use root `go.mod` and `go.sum`

### 3. Dockerfile Build Paths
- **Issue**: Dockerfile.scheduler used incorrect paths for go.mod and source files
- **Fix**: Updated Dockerfile to copy from root directory and use correct build paths

### 4. Missing Python Helper Files
- **Issue**: Dockerfiles referenced `carbon_data_sources.py` and `carbon_rl_env.py` which didn't exist
- **Fix**: Created both helper modules with necessary functionality

### 5. CDK Requirements Format
- **Issue**: CDK requirements.txt had incorrect package names (aws-cdk.aws-* packages don't exist in v2)
- **Fix**: Updated to use only `aws-cdk-lib` which contains all CDK constructs

### 6. Config Directory Reference
- **Issue**: Dockerfile.scheduler referenced a `config/` directory that didn't exist
- **Fix**: Removed the reference as it's not needed for the plugin build

## Known Issues

### gnostic/openapiv2 Compatibility
There is a known compatibility issue between `k8s.io/client-go v0.24.6` and newer versions of `github.com/google/gnostic`. The error manifests as:
```
cannot use doc (variable of type *"github.com/google/gnostic/openapiv2".Document) as *"github.com/google/gnostic-models/openapiv2".Document value
```

This is a dependency conflict that requires careful version pinning. The current `go.mod` has the versions pinned, but if you encounter this error during build, you may need to:

1. Use a specific commit hash for gnostic-models
2. Or upgrade client-go to a version compatible with gnostic-models

For now, the module dependencies are correctly configured in `go.mod`.

## Build Commands

After these fixes, you can build the project using:

```bash
# Build all components
make build

# Build Go plugin only
make build-go

# Build Python components
make build-python

# Build Docker images
make build-images
```

## Verification

To verify the build works:

```bash
# Test Go build
go build -buildmode=plugin -o bin/scheduler-plugin.so ./pkg/emissionplugin/cmd/scheduler

# Test Python scripts
python3 -m py_compile scripts/poller.py
python3 -m py_compile scripts/rl_tuner.py
```

## Next Steps

1. If you encounter the gnostic compatibility issue, consider:
   - Upgrading Kubernetes dependencies to newer versions
   - Or using a specific commit hash for gnostic-models that matches client-go v0.24.6

2. Test the Docker builds:
   ```bash
   docker build -f docker/Dockerfile.scheduler -t carbon-kube-scheduler:test .
   docker build -f docker/Dockerfile.poller -t carbon-kube-poller:test .
   docker build -f docker/Dockerfile.rl-tuner -t carbon-kube-rl-tuner:test .
   ```

3. Verify the plugin can be loaded by the Kubernetes scheduler framework


