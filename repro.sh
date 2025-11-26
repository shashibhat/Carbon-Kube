#!/bin/bash

# Carbon-Kube GPU Evaluation Reproduction Script
# Full deployment and evaluation pipeline (<45min execution)
# 
# This script automates the complete Carbon-Kube GPU evaluation process:
# 1. Environment setup and validation
# 2. Carbon-Kube deployment with GPU support
# 3. GPU workload deployment and execution
# 4. Baseline and ablation study evaluation
# 5. Results collection and analysis

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="carbon-kube-system"
GPU_NAMESPACE="gpu-operator"
WORKLOAD_NAMESPACE="carbon-kube-eval"
RESULTS_DIR="${SCRIPT_DIR}/evaluation/results/$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULTS_DIR}/repro.log"
TIMEOUT_DEPLOYMENT=600  # 10 minutes
TIMEOUT_WORKLOAD=1800   # 30 minutes
PARALLEL_JOBS=4

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "${LOG_FILE}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "${LOG_FILE}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "${LOG_FILE}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "${LOG_FILE}"
}

# Progress tracking
TOTAL_STEPS=12
CURRENT_STEP=0

progress() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo -e "${BLUE}[Step ${CURRENT_STEP}/${TOTAL_STEPS}] $1${NC}" | tee -a "${LOG_FILE}"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        warn "Script failed with exit code $exit_code. Cleaning up..."
        
        # Clean up workloads
        kubectl delete namespace "${WORKLOAD_NAMESPACE}" --ignore-not-found=true --timeout=60s || true
        
        # Clean up Carbon-Kube if requested
        if [ "${CLEANUP_ON_FAILURE:-false}" = "true" ]; then
            helm uninstall carbon-kube -n "${NAMESPACE}" || true
            kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true --timeout=120s || true
        fi
    fi
    
    log "Cleanup completed. Results saved in: ${RESULTS_DIR}"
}

trap cleanup EXIT

# Prerequisites check
check_prerequisites() {
    progress "Checking prerequisites"
    
    # Check required tools
    local tools=("kubectl" "helm" "docker" "python3" "jq" "curl")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
        fi
    done
    
    # Check Kubernetes cluster
    if ! kubectl cluster-info &> /dev/null; then
        error "Kubernetes cluster is not accessible"
    fi
    
    # Check GPU nodes
    local gpu_nodes=$(kubectl get nodes -l accelerator=nvidia --no-headers 2>/dev/null | wc -l)
    if [ "$gpu_nodes" -eq 0 ]; then
        warn "No GPU nodes found. Looking for nodes with nvidia.com/gpu label..."
        gpu_nodes=$(kubectl get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l)
        if [ "$gpu_nodes" -eq 0 ]; then
            error "No GPU nodes found in the cluster. Please ensure GPU nodes are available."
        fi
    fi
    
    log "Found $gpu_nodes GPU node(s)"
    
    # Check Helm repositories
    if ! helm repo list | grep -q "prometheus-community"; then
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    fi
    if ! helm repo list | grep -q "grafana"; then
        helm repo add grafana https://grafana.github.io/helm-charts
    fi
    if ! helm repo list | grep -q "nvidia"; then
        helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
    fi
    
    helm repo update
    
    log "Prerequisites check completed"
}

# Environment setup
setup_environment() {
    progress "Setting up environment"
    
    # Create results directory
    mkdir -p "${RESULTS_DIR}"
    
    # Create namespaces
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace "${WORKLOAD_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespaces
    kubectl label namespace "${NAMESPACE}" carbon-kube.io/system=true --overwrite
    kubectl label namespace "${WORKLOAD_NAMESPACE}" carbon-kube.io/evaluation=true --overwrite
    
    # Setup Python environment for evaluation
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install -r evaluation/requirements.txt
    
    log "Environment setup completed"
}

# Build and deploy Carbon-Kube
deploy_carbon_kube() {
    progress "Building and deploying Carbon-Kube"
    
    # Build Docker images
    log "Building Docker images..."
    make build-scheduler &
    make build-poller &
    make build-rl-tuner &
    wait
    
    # Deploy with GPU support enabled
    log "Deploying Carbon-Kube with GPU support..."
    helm upgrade --install carbon-kube ./charts/carbon-kube \
        --namespace "${NAMESPACE}" \
        --set gpu.nvidia.enabled=true \
        --set gpu.monitoring.dcgm.enabled=true \
        --set gpu.workloads.scheduling.carbonAware=true \
        --set monitoring.prometheus.enabled=true \
        --set monitoring.grafana.enabled=true \
        --timeout "${TIMEOUT_DEPLOYMENT}s" \
        --wait
    
    # Wait for all pods to be ready
    log "Waiting for Carbon-Kube pods to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=carbon-kube \
        -n "${NAMESPACE}" --timeout="${TIMEOUT_DEPLOYMENT}s"
    
    # Verify DCGM exporter is running
    if kubectl get pods -n "${GPU_NAMESPACE}" -l app=dcgm-exporter --no-headers 2>/dev/null | grep -q Running; then
        log "DCGM exporter is running"
    else
        warn "DCGM exporter not found, GPU metrics may be limited"
    fi
    
    log "Carbon-Kube deployment completed"
}

# Deploy GPU workloads
deploy_gpu_workloads() {
    progress "Deploying GPU workloads"
    
    local workloads=("bert-finetune" "llama-inference" "rapids-tpcds" "flink-nexmark")
    
    for workload in "${workloads[@]}"; do
        log "Deploying $workload workload..."
        kubectl apply -f "test/workloads/gpu/${workload}.yaml" -n "${WORKLOAD_NAMESPACE}"
    done
    
    # Wait for workloads to be scheduled
    log "Waiting for workloads to be scheduled..."
    sleep 30
    
    # Check workload status
    for workload in "${workloads[@]}"; do
        local status=$(kubectl get pods -n "${WORKLOAD_NAMESPACE}" -l workload="$workload" \
            --no-headers 2>/dev/null | awk '{print $3}' | head -1)
        log "$workload status: ${status:-Not Found}"
    done
    
    log "GPU workloads deployment completed"
}

# Run baseline evaluation
run_baseline_evaluation() {
    progress "Running baseline evaluation (HPA + Static Scheduling)"
    
    log "Configuring baseline schedulers..."
    
    # Disable Carbon-Kube scheduler temporarily
    kubectl patch deployment carbon-kube-scheduler -n "${NAMESPACE}" \
        -p '{"spec":{"replicas":0}}'
    
    # Deploy HPA for baseline
    cat <<EOF | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: baseline-hpa
  namespace: ${WORKLOAD_NAMESPACE}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-inference
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
EOF
    
    # Run baseline workloads for 10 minutes
    log "Running baseline evaluation for 10 minutes..."
    local baseline_start=$(date +%s)
    
    # Collect baseline metrics
    kubectl top nodes > "${RESULTS_DIR}/baseline_nodes_$(date +%H%M%S).txt" &
    kubectl top pods -n "${WORKLOAD_NAMESPACE}" > "${RESULTS_DIR}/baseline_pods_$(date +%H%M%S).txt" &
    
    # Monitor GPU metrics if DCGM is available
    if kubectl get svc dcgm-exporter -n "${GPU_NAMESPACE}" &>/dev/null; then
        kubectl port-forward -n "${GPU_NAMESPACE}" svc/dcgm-exporter 9400:9400 &
        local port_forward_pid=$!
        sleep 5
        curl -s http://localhost:9400/metrics | grep -E "(DCGM_FI_DEV_POWER_USAGE|DCGM_FI_DEV_GPU_UTIL)" \
            > "${RESULTS_DIR}/baseline_gpu_metrics_$(date +%H%M%S).txt" &
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    sleep 600  # 10 minutes
    
    local baseline_end=$(date +%s)
    local baseline_duration=$((baseline_end - baseline_start))
    
    log "Baseline evaluation completed in ${baseline_duration}s"
    
    # Re-enable Carbon-Kube scheduler
    kubectl patch deployment carbon-kube-scheduler -n "${NAMESPACE}" \
        -p '{"spec":{"replicas":1}}'
    
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=carbon-kube-scheduler \
        -n "${NAMESPACE}" --timeout=120s
}

# Run ablation studies
run_ablation_studies() {
    progress "Running ablation studies"
    
    local studies=("no-rl" "no-mig" "no-dcgm")
    
    for study in "${studies[@]}"; do
        log "Running ablation study: $study"
        
        case $study in
            "no-rl")
                # Disable RL tuner
                kubectl patch cronjob carbon-kube-rl-tuner -n "${NAMESPACE}" \
                    -p '{"spec":{"suspend":true}}'
                ;;
            "no-mig")
                # Disable MIG
                helm upgrade carbon-kube ./charts/carbon-kube \
                    --namespace "${NAMESPACE}" \
                    --reuse-values \
                    --set gpu.mig.enabled=false \
                    --timeout 300s
                ;;
            "no-dcgm")
                # Disable DCGM
                helm upgrade carbon-kube ./charts/carbon-kube \
                    --namespace "${NAMESPACE}" \
                    --reuse-values \
                    --set gpu.monitoring.dcgm.enabled=false \
                    --timeout 300s
                ;;
        esac
        
        # Wait for changes to take effect
        sleep 60
        
        # Run workloads for 5 minutes
        local study_start=$(date +%s)
        
        # Restart workloads to trigger rescheduling
        kubectl rollout restart deployment -n "${WORKLOAD_NAMESPACE}" || true
        kubectl delete jobs --all -n "${WORKLOAD_NAMESPACE}" || true
        kubectl apply -f test/workloads/gpu/ -n "${WORKLOAD_NAMESPACE}"
        
        sleep 300  # 5 minutes
        
        # Collect metrics
        kubectl top nodes > "${RESULTS_DIR}/ablation_${study}_nodes_$(date +%H%M%S).txt"
        kubectl top pods -n "${WORKLOAD_NAMESPACE}" > "${RESULTS_DIR}/ablation_${study}_pods_$(date +%H%M%S).txt"
        
        local study_end=$(date +%s)
        local study_duration=$((study_end - study_start))
        
        log "Ablation study $study completed in ${study_duration}s"
    done
    
    # Restore full configuration
    log "Restoring full Carbon-Kube configuration..."
    helm upgrade carbon-kube ./charts/carbon-kube \
        --namespace "${NAMESPACE}" \
        -f charts/carbon-kube/charts/nvidia-operator/values-gpu.yaml \
        --timeout 300s
    
    kubectl patch cronjob carbon-kube-rl-tuner -n "${NAMESPACE}" \
        -p '{"spec":{"suspend":false}}'
}

# Run full Carbon-Kube evaluation
run_carbon_kube_evaluation() {
    progress "Running full Carbon-Kube GPU evaluation"
    
    log "Starting comprehensive Carbon-Kube evaluation..."
    
    # Ensure all components are running
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=carbon-kube \
        -n "${NAMESPACE}" --timeout=300s
    
    # Run evaluation for 15 minutes with full Carbon-Kube features
    local eval_start=$(date +%s)
    
    # Deploy all GPU workloads with carbon-aware scheduling
    kubectl apply -f test/workloads/gpu/ -n "${WORKLOAD_NAMESPACE}"
    
    # Monitor and collect metrics every minute
    for i in {1..15}; do
        log "Carbon-Kube evaluation: minute $i/15"
        
        # Collect comprehensive metrics
        kubectl top nodes > "${RESULTS_DIR}/carbon_kube_nodes_${i}min.txt"
        kubectl top pods -n "${WORKLOAD_NAMESPACE}" > "${RESULTS_DIR}/carbon_kube_pods_${i}min.txt"
        
        # Collect Carbon-Kube specific metrics
        kubectl logs -l app.kubernetes.io/name=carbon-kube-scheduler -n "${NAMESPACE}" \
            --tail=100 > "${RESULTS_DIR}/carbon_kube_scheduler_${i}min.log"
        
        # Collect GPU metrics if available
        if kubectl get svc dcgm-exporter -n "${GPU_NAMESPACE}" &>/dev/null; then
            kubectl port-forward -n "${GPU_NAMESPACE}" svc/dcgm-exporter 9400:9400 &
            local port_forward_pid=$!
            sleep 2
            curl -s http://localhost:9400/metrics | grep -E "(DCGM_FI_DEV_POWER_USAGE|DCGM_FI_DEV_GPU_UTIL|DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION)" \
                > "${RESULTS_DIR}/carbon_kube_gpu_${i}min.txt" 2>/dev/null || true
            kill $port_forward_pid 2>/dev/null || true
        fi
        
        sleep 60
    done
    
    local eval_end=$(date +%s)
    local eval_duration=$((eval_end - eval_start))
    
    log "Carbon-Kube evaluation completed in ${eval_duration}s"
}

# Generate comparison report
generate_report() {
    progress "Generating comparison report"
    
    log "Analyzing results and generating report..."
    
    # Activate Python environment
    source venv/bin/activate
    
    # Run Jupyter notebook for analysis
    log "Running GPU Carbon Evaluation notebook..."
    cd evaluation/notebooks
    
    # Convert notebook to script and run
    jupyter nbconvert --to script 05_GPU_Carbon_Evaluation.ipynb
    python 05_GPU_Carbon_Evaluation.py "${RESULTS_DIR}" > "${RESULTS_DIR}/analysis.log" 2>&1 || true
    
    cd "${SCRIPT_DIR}"
    
    # Generate summary report
    cat > "${RESULTS_DIR}/summary_report.md" << EOF
# Carbon-Kube GPU Evaluation Summary Report

**Generated:** $(date)
**Duration:** Total evaluation time
**GPU Nodes:** $(kubectl get nodes -l accelerator=nvidia --no-headers | wc -l)

## Evaluation Components

### 1. Baseline Evaluation (HPA + Static Scheduling)
- Duration: 10 minutes
- Scheduler: Kubernetes default + HPA
- GPU Management: Static allocation

### 2. Ablation Studies
- **No RL**: Carbon-Kube without RL tuner
- **No MIG**: Carbon-Kube without MIG support  
- **No DCGM**: Carbon-Kube without DCGM metrics
- Duration: 5 minutes each

### 3. Full Carbon-Kube Evaluation
- Duration: 15 minutes
- Features: All Carbon-Kube features enabled
- GPU Management: DCGM + MIG + Carbon-aware scheduling

## GPU Workloads Evaluated

1. **BERT Fine-tuning**: NLP model training workload
2. **Llama Inference**: Large language model inference with vLLM
3. **RAPIDS TPC-DS**: GPU-accelerated analytics workload
4. **Flink NexMark**: Stream processing with GPU acceleration

## Key Metrics Collected

- **Power Consumption**: Real-time GPU power draw (DCGM)
- **GPU Utilization**: Compute and memory utilization
- **Energy Consumption**: Cumulative energy usage
- **Carbon Emissions**: CO₂ equivalent based on grid intensity
- **Latency**: Workload completion times
- **Throughput**: Requests/operations per second

## Results Location

All raw data and analysis results are stored in:
\`${RESULTS_DIR}\`

## Next Steps

1. Review detailed analysis in \`05_GPU_Carbon_Evaluation.ipynb\`
2. Examine raw metrics in individual result files
3. Compare CO₂ vs latency trade-offs across configurations
4. Validate statistical significance of improvements

EOF
    
    log "Report generation completed"
}

# Collect final results
collect_results() {
    progress "Collecting final results"
    
    # Collect cluster state
    kubectl get nodes -o wide > "${RESULTS_DIR}/final_cluster_nodes.txt"
    kubectl get pods --all-namespaces > "${RESULTS_DIR}/final_cluster_pods.txt"
    
    # Collect Carbon-Kube configuration
    helm get values carbon-kube -n "${NAMESPACE}" > "${RESULTS_DIR}/carbon_kube_config.yaml"
    
    # Collect GPU operator status
    if kubectl get namespace "${GPU_NAMESPACE}" &>/dev/null; then
        kubectl get pods -n "${GPU_NAMESPACE}" > "${RESULTS_DIR}/gpu_operator_pods.txt"
    fi
    
    # Archive logs
    kubectl logs -l app.kubernetes.io/name=carbon-kube-scheduler -n "${NAMESPACE}" \
        > "${RESULTS_DIR}/carbon_kube_scheduler_full.log" || true
    kubectl logs -l app.kubernetes.io/name=carbon-kube-poller -n "${NAMESPACE}" \
        > "${RESULTS_DIR}/carbon_kube_poller_full.log" || true
    
    # Create results archive
    tar -czf "${RESULTS_DIR}.tar.gz" -C "$(dirname "${RESULTS_DIR}")" "$(basename "${RESULTS_DIR}")"
    
    log "Results collected and archived: ${RESULTS_DIR}.tar.gz"
}

# Cleanup workloads
cleanup_workloads() {
    progress "Cleaning up evaluation workloads"
    
    # Delete evaluation namespace
    kubectl delete namespace "${WORKLOAD_NAMESPACE}" --timeout=120s || true
    
    # Clean up HPA
    kubectl delete hpa baseline-hpa -n "${WORKLOAD_NAMESPACE}" --ignore-not-found=true || true
    
    log "Workload cleanup completed"
}

# Main execution
main() {
    local start_time=$(date +%s)
    
    log "Starting Carbon-Kube GPU Evaluation Reproduction"
    log "Results will be saved to: ${RESULTS_DIR}"
    
    # Execute all steps
    check_prerequisites
    setup_environment
    deploy_carbon_kube
    deploy_gpu_workloads
    run_baseline_evaluation
    run_ablation_studies
    run_carbon_kube_evaluation
    generate_report
    collect_results
    cleanup_workloads
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local minutes=$((total_duration / 60))
    local seconds=$((total_duration % 60))
    
    log "Carbon-Kube GPU Evaluation completed successfully!"
    log "Total execution time: ${minutes}m ${seconds}s"
    log "Results available at: ${RESULTS_DIR}"
    
    # Final summary
    echo
    echo "=========================================="
    echo "  Carbon-Kube GPU Evaluation Complete"
    echo "=========================================="
    echo "Execution time: ${minutes}m ${seconds}s"
    echo "Results: ${RESULTS_DIR}"
    echo "Archive: ${RESULTS_DIR}.tar.gz"
    echo
    echo "Next steps:"
    echo "1. Review summary report: ${RESULTS_DIR}/summary_report.md"
    echo "2. Analyze detailed results: evaluation/notebooks/05_GPU_Carbon_Evaluation.ipynb"
    echo "3. Examine raw metrics in results directory"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cleanup-on-failure)
            CLEANUP_ON_FAILURE=true
            shift
            ;;
        --timeout-deployment)
            TIMEOUT_DEPLOYMENT="$2"
            shift 2
            ;;
        --timeout-workload)
            TIMEOUT_WORKLOAD="$2"
            shift 2
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --cleanup-on-failure    Clean up Carbon-Kube on script failure"
            echo "  --timeout-deployment N  Deployment timeout in seconds (default: 600)"
            echo "  --timeout-workload N    Workload timeout in seconds (default: 1800)"
            echo "  --parallel-jobs N       Number of parallel jobs (default: 4)"
            echo "  --help                  Show this help message"
            echo
            echo "This script runs a complete Carbon-Kube GPU evaluation including:"
            echo "- Environment setup and validation"
            echo "- Carbon-Kube deployment with GPU support"
            echo "- Baseline evaluation (HPA + static scheduling)"
            echo "- Ablation studies (no-RL, no-MIG, no-DCGM)"
            echo "- Full Carbon-Kube evaluation"
            echo "- Results analysis and reporting"
            echo
            echo "Expected execution time: <45 minutes"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run main function
main "$@"