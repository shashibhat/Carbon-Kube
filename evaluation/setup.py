#!/usr/bin/env python3
"""
Carbon-Kube Evaluation Framework Setup Script
Advanced configuration and package management
"""

import os
import sys
import subprocess
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationSetup:
    """Setup manager for the evaluation framework"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.evaluation_path = self.base_path / "evaluation"
        self.config = {}
        
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        logger.info("Checking system requirements...")
        
        requirements = {
            "python": "3.8",
            "go": "1.19",
            "git": "2.0"
        }
        
        for tool, min_version in requirements.items():
            if not self._check_tool_version(tool, min_version):
                logger.error(f"{tool} version {min_version}+ required")
                return False
                
        logger.info("System requirements satisfied")
        return True
    
    def _check_tool_version(self, tool: str, min_version: str) -> bool:
        """Check if a tool meets minimum version requirement"""
        try:
            if tool == "python":
                version = f"{sys.version_info.major}.{sys.version_info.minor}"
            elif tool == "go":
                result = subprocess.run(["go", "version"], capture_output=True, text=True)
                version = result.stdout.split()[2].replace("go", "")
            elif tool == "git":
                result = subprocess.run(["git", "--version"], capture_output=True, text=True)
                version = result.stdout.split()[2]
            else:
                return False
                
            return self._version_compare(version, min_version) >= 0
        except Exception as e:
            logger.warning(f"Could not check {tool} version: {e}")
            return False
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare two version strings"""
        v1_parts = [int(x) for x in version1.split('.')]
        v2_parts = [int(x) for x in version2.split('.')]
        
        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))
        
        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0
    
    def create_directory_structure(self):
        """Create the evaluation framework directory structure"""
        logger.info("Creating directory structure...")
        
        directories = [
            "evaluation/data/raw",
            "evaluation/data/processed", 
            "evaluation/data/synthetic",
            "evaluation/data/benchmarks",
            "evaluation/notebooks",
            "evaluation/results/baselines",
            "evaluation/results/experiments",
            "evaluation/results/ablations",
            "evaluation/results/reports",
            "evaluation/artifacts/experiments",
            "evaluation/artifacts/models",
            "evaluation/artifacts/datasets",
            "evaluation/configs",
            "evaluation/scripts",
            "evaluation/tests",
            "evaluation/docs",
            "evaluation/logs",
            "evaluation/bin"
        ]
        
        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info("Directory structure created")
    
    def install_python_packages(self):
        """Install required Python packages"""
        logger.info("Installing Python packages...")
        
        packages = [
            "numpy>=1.21.0",
            "pandas>=1.3.0", 
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
            "prometheus-client>=0.12.0",
            "kubernetes>=18.0.0",
            "pyyaml>=5.4.0",
            "requests>=2.25.0",
            "psutil>=5.8.0",
            "tqdm>=4.62.0",
            "click>=8.0.0",
            "rich>=10.0.0",
            "tabulate>=0.8.0"
        ]
        
        for package in packages:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                logger.info(f"Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {package}: {e}")
                
        logger.info("Python packages installation completed")
    
    def create_configuration_files(self):
        """Create configuration files"""
        logger.info("Creating configuration files...")
        
        # Main evaluation config
        eval_config = {
            "framework": {
                "name": "Carbon-Kube Evaluation",
                "version": "1.0.0",
                "description": "Scientific evaluation framework for carbon-efficient Kubernetes scheduling"
            },
            "statistical_analysis": {
                "confidence_level": 0.95,
                "significance_threshold": 0.05,
                "multiple_comparisons_correction": "bonferroni",
                "bootstrap_samples": 1000,
                "effect_size_threshold": 0.2
            },
            "metrics": {
                "primary_metrics": [
                    "carbon_efficiency",
                    "energy_consumption", 
                    "performance_score"
                ],
                "secondary_metrics": [
                    "resource_utilization",
                    "response_time",
                    "throughput",
                    "queue_time",
                    "completion_rate"
                ],
                "aggregation_methods": ["mean", "median", "std", "min", "max"]
            },
            "reproducibility": {
                "capture_environment": True,
                "track_dependencies": True,
                "store_artifacts": True,
                "validate_checksums": True,
                "seed_management": True,
                "version_control": True
            },
            "reporting": {
                "generate_plots": True,
                "include_raw_data": False,
                "format": "html",
                "template": "default",
                "auto_open": False
            },
            "storage": {
                "base_path": "./evaluation/artifacts",
                "compression": True,
                "encryption": False,
                "backup": True,
                "retention_days": 30
            }
        }
        
        config_path = self.evaluation_path / "configs" / "evaluation.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(eval_config, f, default_flow_style=False, indent=2)
        
        # Logging configuration
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "evaluation/logs/evaluation.log",
                    "maxBytes": 10485760,
                    "backupCount": 5
                }
            },
            "loggers": {
                "evaluation": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["console"]
            }
        }
        
        logging_path = self.evaluation_path / "configs" / "logging.yaml"
        with open(logging_path, 'w') as f:
            yaml.dump(logging_config, f, default_flow_style=False, indent=2)
        
        logger.info("Configuration files created")
    
    def create_environment_file(self):
        """Create environment configuration file"""
        logger.info("Creating environment file...")
        
        env_content = """# Carbon-Kube Evaluation Environment Variables

# Framework Configuration
EVALUATION_CONFIG_PATH=./evaluation/configs/evaluation.yaml
EVALUATION_DATA_PATH=./evaluation/data
EVALUATION_RESULTS_PATH=./evaluation/results
EVALUATION_ARTIFACTS_PATH=./evaluation/artifacts

# Logging
LOG_LEVEL=INFO
LOG_CONFIG_PATH=./evaluation/configs/logging.yaml

# Python Environment
PYTHONPATH=./evaluation:$PYTHONPATH

# Kubernetes Configuration
KUBECONFIG=${KUBECONFIG:-~/.kube/config}
KUBERNETES_NAMESPACE=${KUBERNETES_NAMESPACE:-default}

# Monitoring
PROMETHEUS_URL=${PROMETHEUS_URL:-http://localhost:9090}
GRAFANA_URL=${GRAFANA_URL:-http://localhost:3000}

# GPU Configuration
NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}

# Experiment Configuration
EXPERIMENT_TIMEOUT=3600
MAX_CONCURRENT_EXPERIMENTS=4
ARTIFACT_RETENTION_DAYS=30

# Security
EVALUATION_SECRET_KEY=${EVALUATION_SECRET_KEY:-change-me-in-production}
"""
        
        env_path = self.evaluation_path / ".env"
        with open(env_path, 'w') as f:
            f.write(env_content)
            
        logger.info("Environment file created")
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        logger.info("Creating requirements.txt...")
        
        requirements = [
            "# Core scientific computing",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "",
            "# Visualization",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0", 
            "plotly>=5.0.0",
            "",
            "# Jupyter ecosystem",
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
            "",
            "# Monitoring and metrics",
            "prometheus-client>=0.12.0",
            "",
            "# Kubernetes integration",
            "kubernetes>=18.0.0",
            "",
            "# Utilities",
            "pyyaml>=5.4.0",
            "requests>=2.25.0",
            "psutil>=5.8.0",
            "tqdm>=4.62.0",
            "click>=8.0.0",
            "rich>=10.0.0",
            "tabulate>=0.8.0",
            "",
            "# Testing",
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "pytest-mock>=3.6.0"
        ]
        
        req_path = self.evaluation_path / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
            
        logger.info("Requirements file created")
    
    def create_makefile(self):
        """Create Makefile for common tasks"""
        logger.info("Creating Makefile...")
        
        makefile_content = """# Carbon-Kube Evaluation Framework Makefile

.PHONY: help setup install test clean build run-jupyter run-tests lint format

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

setup: ## Run initial setup
	@echo "Setting up evaluation framework..."
	chmod +x evaluation/setup.sh
	./evaluation/setup.sh

install: ## Install Python dependencies
	pip install -r evaluation/requirements.txt

test: ## Run all tests
	@echo "Running Go tests..."
	go test ./evaluation/... -v
	@echo "Running Python tests..."
	cd evaluation && python -m pytest tests/ -v

clean: ## Clean build artifacts and temporary files
	@echo "Cleaning up..."
	rm -rf evaluation/bin/*
	rm -rf evaluation/__pycache__
	rm -rf evaluation/*/__pycache__
	rm -rf evaluation/.pytest_cache
	find evaluation -name "*.pyc" -delete
	find evaluation -name "*.pyo" -delete

build: ## Build Go binaries
	@echo "Building Go binaries..."
	mkdir -p evaluation/bin
	go build -o evaluation/bin/evaluation-framework ./evaluation/

run-jupyter: ## Start Jupyter Lab
	@echo "Starting Jupyter Lab..."
	cd evaluation && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

run-tests: ## Run tests with coverage
	@echo "Running tests with coverage..."
	cd evaluation && python -m pytest tests/ -v --cov=. --cov-report=html

lint: ## Run linters
	@echo "Running linters..."
	cd evaluation && python -m flake8 .
	go vet ./evaluation/...

format: ## Format code
	@echo "Formatting code..."
	cd evaluation && python -m black .
	go fmt ./evaluation/...

docs: ## Generate documentation
	@echo "Generating documentation..."
	cd evaluation && python -m pydoc -w .

.DEFAULT_GOAL := help
"""
        
        makefile_path = self.base_path / "Makefile.evaluation"
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
            
        logger.info("Makefile created")
    
    def validate_setup(self) -> bool:
        """Validate the setup"""
        logger.info("Validating setup...")
        
        required_files = [
            "evaluation/configs/evaluation.yaml",
            "evaluation/configs/logging.yaml",
            "evaluation/.env",
            "evaluation/requirements.txt"
        ]
        
        required_dirs = [
            "evaluation/data",
            "evaluation/notebooks", 
            "evaluation/results",
            "evaluation/artifacts",
            "evaluation/configs",
            "evaluation/scripts",
            "evaluation/tests",
            "evaluation/docs",
            "evaluation/logs"
        ]
        
        # Check files
        for file_path in required_files:
            if not (self.base_path / file_path).exists():
                logger.error(f"Missing required file: {file_path}")
                return False
                
        # Check directories
        for dir_path in required_dirs:
            if not (self.base_path / dir_path).is_dir():
                logger.error(f"Missing required directory: {dir_path}")
                return False
                
        logger.info("Setup validation passed")
        return True
    
    def run_setup(self):
        """Run the complete setup process"""
        logger.info("Starting Carbon-Kube Evaluation Framework setup...")
        
        try:
            if not self.check_system_requirements():
                sys.exit(1)
                
            self.create_directory_structure()
            self.install_python_packages()
            self.create_configuration_files()
            self.create_environment_file()
            self.create_requirements_file()
            self.create_makefile()
            
            if self.validate_setup():
                logger.info("Setup completed successfully! 🚀")
                self._print_next_steps()
            else:
                logger.error("Setup validation failed")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            sys.exit(1)
    
    def _print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*60)
        print("🎉 Carbon-Kube Evaluation Framework Setup Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Activate environment variables:")
        print("   export $(cat evaluation/.env | xargs)")
        print("\n2. Start Jupyter Lab:")
        print("   make -f Makefile.evaluation run-jupyter")
        print("\n3. Run tests:")
        print("   make -f Makefile.evaluation test")
        print("\n4. Check documentation:")
        print("   ls evaluation/docs/")
        print("\n5. Start evaluating:")
        print("   cd evaluation/notebooks/")
        print("\nHappy evaluating! 🚀")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Carbon-Kube Evaluation Framework Setup")
    parser.add_argument("--base-path", default=".", help="Base path for setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup = EvaluationSetup(args.base_path)
    setup.run_setup()

if __name__ == "__main__":
    main()