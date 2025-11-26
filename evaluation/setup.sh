#!/bin/bash

# Carbon-Kube Evaluation Framework Setup Script
# This script sets up the environment for running scientific evaluations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on macOS or Linux
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    log_info "Detected OS: $OS"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Go installation
    if ! command -v go &> /dev/null; then
        log_error "Go is not installed. Please install Go 1.19 or later."
        exit 1
    fi
    
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    log_success "Go version: $GO_VERSION"
    
    # Check Python installation
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.8 or later."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_success "Python version: $PYTHON_VERSION"
    
    # Check pip installation
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed. Please install pip3."
        exit 1
    fi
    
    # Check Docker (optional but recommended)
    if command -v docker &> /dev/null; then
        log_success "Docker is available"
    else
        log_warning "Docker is not installed. Some features may not work."
    fi
    
    # Check kubectl (optional)
    if command -v kubectl &> /dev/null; then
        log_success "kubectl is available"
    else
        log_warning "kubectl is not installed. Kubernetes features will not work."
    fi
}

# Install Go dependencies
install_go_dependencies() {
    log_info "Installing Go dependencies..."
    
    # Initialize Go module if not exists
    if [ ! -f "go.mod" ]; then
        log_info "Initializing Go module..."
        go mod init carbon-kube-evaluation
    fi
    
    # Install required Go packages
    log_info "Installing Go packages..."
    go get -u github.com/prometheus/client_golang/prometheus
    go get -u github.com/prometheus/client_golang/prometheus/promhttp
    go get -u gopkg.in/yaml.v2
    go get -u github.com/sirupsen/logrus
    go get -u github.com/stretchr/testify/assert
    go get -u github.com/gorilla/mux
    go get -u k8s.io/client-go/kubernetes
    go get -u k8s.io/apimachinery/pkg/apis/meta/v1
    
    # Tidy up dependencies
    go mod tidy
    
    log_success "Go dependencies installed successfully"
}

# Install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install required packages
    log_info "Installing Python packages..."
    pip install numpy>=1.21.0
    pip install pandas>=1.3.0
    pip install scipy>=1.7.0
    pip install scikit-learn>=1.0.0
    pip install matplotlib>=3.4.0
    pip install seaborn>=0.11.0
    pip install jupyter>=1.0.0
    pip install jupyterlab>=3.0.0
    pip install plotly>=5.0.0
    pip install prometheus-client>=0.12.0
    pip install kubernetes>=18.0.0
    pip install pyyaml>=5.4.0
    pip install requests>=2.25.0
    pip install psutil>=5.8.0
    
    # Create requirements.txt
    pip freeze > evaluation/requirements.txt
    
    log_success "Python dependencies installed successfully"
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    # Create evaluation subdirectories
    mkdir -p evaluation/data
    mkdir -p evaluation/notebooks
    mkdir -p evaluation/results
    mkdir -p evaluation/artifacts
    mkdir -p evaluation/configs
    mkdir -p evaluation/scripts
    mkdir -p evaluation/tests
    mkdir -p evaluation/docs
    
    # Create data subdirectories
    mkdir -p evaluation/data/raw
    mkdir -p evaluation/data/processed
    mkdir -p evaluation/data/synthetic
    mkdir -p evaluation/data/benchmarks
    
    # Create results subdirectories
    mkdir -p evaluation/results/baselines
    mkdir -p evaluation/results/experiments
    mkdir -p evaluation/results/ablations
    mkdir -p evaluation/results/reports
    
    log_success "Directory structure created"
}

# Create configuration files
create_config_files() {
    log_info "Creating configuration files..."
    
    # Create evaluation config
    cat > evaluation/configs/evaluation.yaml << EOF
# Evaluation Framework Configuration
framework:
  name: "Carbon-Kube Evaluation"
  version: "1.0.0"
  
statistical_analysis:
  confidence_level: 0.95
  significance_threshold: 0.05
  multiple_comparisons_correction: "bonferroni"
  bootstrap_samples: 1000
  
metrics:
  primary_metrics:
    - "carbon_efficiency"
    - "energy_consumption"
    - "performance_score"
  secondary_metrics:
    - "resource_utilization"
    - "response_time"
    - "throughput"
  
reproducibility:
  capture_environment: true
  track_dependencies: true
  store_artifacts: true
  validate_checksums: true
  
reporting:
  generate_plots: true
  include_raw_data: false
  format: "html"
  
storage:
  base_path: "./evaluation/artifacts"
  compression: true
  encryption: false
EOF

    # Create logging config
    cat > evaluation/configs/logging.yaml << EOF
# Logging Configuration
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - type: "console"
    - type: "file"
      filename: "evaluation/logs/evaluation.log"
      max_bytes: 10485760  # 10MB
      backup_count: 5
EOF

    log_success "Configuration files created"
}

# Set up environment variables
setup_environment() {
    log_info "Setting up environment variables..."
    
    # Create .env file
    cat > evaluation/.env << EOF
# Carbon-Kube Evaluation Environment Variables

# Framework Configuration
EVALUATION_CONFIG_PATH=./evaluation/configs/evaluation.yaml
EVALUATION_DATA_PATH=./evaluation/data
EVALUATION_RESULTS_PATH=./evaluation/results
EVALUATION_ARTIFACTS_PATH=./evaluation/artifacts

# Logging
LOG_LEVEL=INFO
LOG_FILE=./evaluation/logs/evaluation.log

# Kubernetes (if available)
KUBECONFIG=${KUBECONFIG:-~/.kube/config}

# Prometheus (if available)
PROMETHEUS_URL=${PROMETHEUS_URL:-http://localhost:9090}

# GPU Monitoring (if available)
NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}
EOF

    # Create logs directory
    mkdir -p evaluation/logs
    
    log_success "Environment variables configured"
}

# Build Go binaries
build_binaries() {
    log_info "Building Go binaries..."
    
    # Create build directory
    mkdir -p evaluation/bin
    
    # Build evaluation framework binary (if main.go exists)
    if [ -f "evaluation/main.go" ]; then
        go build -o evaluation/bin/evaluation-framework evaluation/main.go
        log_success "Built evaluation-framework binary"
    fi
    
    # Build test binaries
    go test -c -o evaluation/bin/evaluation-tests ./evaluation/...
    
    log_success "Go binaries built successfully"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Run Go tests
    if go test ./evaluation/... -v; then
        log_success "Go tests passed"
    else
        log_warning "Some Go tests failed"
    fi
    
    # Run Python tests (if they exist)
    if [ -d "evaluation/tests" ] && [ -n "$(find evaluation/tests -name '*.py')" ]; then
        source venv/bin/activate
        python -m pytest evaluation/tests/ -v
        log_success "Python tests completed"
    fi
}

# Create example scripts
create_example_scripts() {
    log_info "Creating example scripts..."
    
    # Create quick start script
    cat > evaluation/scripts/quickstart.sh << 'EOF'
#!/bin/bash
# Quick start script for Carbon-Kube evaluation

echo "Starting Carbon-Kube evaluation framework..."

# Activate Python environment
source venv/bin/activate

# Set environment variables
export $(cat evaluation/.env | xargs)

# Run a simple evaluation
echo "Running example evaluation..."
# Add your evaluation commands here

echo "Evaluation completed. Check results in evaluation/results/"
EOF

    chmod +x evaluation/scripts/quickstart.sh
    
    log_success "Example scripts created"
}

# Main setup function
main() {
    log_info "Starting Carbon-Kube Evaluation Framework setup..."
    
    detect_os
    check_prerequisites
    create_directories
    install_go_dependencies
    install_python_dependencies
    create_config_files
    setup_environment
    build_binaries
    create_example_scripts
    
    # Run tests (optional)
    if [[ "${1:-}" == "--with-tests" ]]; then
        run_tests
    fi
    
    log_success "Setup completed successfully!"
    echo
    log_info "Next steps:"
    echo "  1. Activate Python environment: source venv/bin/activate"
    echo "  2. Set environment variables: export \$(cat evaluation/.env | xargs)"
    echo "  3. Run quick start: ./evaluation/scripts/quickstart.sh"
    echo "  4. Check documentation in evaluation/docs/"
    echo
    log_info "Happy evaluating! 🚀"
}

# Run main function with all arguments
main "$@"