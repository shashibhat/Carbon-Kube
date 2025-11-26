#!/usr/bin/env python3
"""
Generate synthetic datasets for Carbon-Kube evaluation framework testing
"""

import numpy as np
import pandas as pd
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random

class DatasetGenerator:
    """Generate synthetic datasets for evaluation framework testing"""
    
    def __init__(self, base_path: str = "./evaluation/data"):
        self.base_path = Path(base_path)
        self.synthetic_path = self.base_path / "synthetic"
        self.benchmarks_path = self.base_path / "benchmarks"
        
        # Create directories
        self.synthetic_path.mkdir(parents=True, exist_ok=True)
        self.benchmarks_path.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
    
    def generate_carbon_efficiency_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic carbon efficiency data"""
        
        # Scheduler types
        schedulers = ["default", "carbon-aware", "energy-efficient", "performance-first"]
        
        # Node types
        node_types = ["cpu-optimized", "memory-optimized", "gpu-enabled", "general-purpose"]
        
        # Workload types
        workload_types = ["batch", "interactive", "ml-training", "web-service"]
        
        data = []
        
        for i in range(n_samples):
            scheduler = np.random.choice(schedulers)
            node_type = np.random.choice(node_types)
            workload_type = np.random.choice(workload_types)
            
            # Base metrics with realistic correlations
            cpu_utilization = np.random.beta(2, 5) * 100  # Skewed towards lower utilization
            memory_utilization = np.random.beta(2, 3) * 100
            
            # Carbon efficiency depends on scheduler and utilization
            if scheduler == "carbon-aware":
                carbon_efficiency = 0.8 + 0.15 * np.random.random()
                energy_consumption = 50 + 30 * (1 - carbon_efficiency) + 10 * np.random.random()
            elif scheduler == "energy-efficient":
                carbon_efficiency = 0.7 + 0.2 * np.random.random()
                energy_consumption = 45 + 35 * (1 - carbon_efficiency) + 15 * np.random.random()
            elif scheduler == "performance-first":
                carbon_efficiency = 0.4 + 0.3 * np.random.random()
                energy_consumption = 80 + 40 * (1 - carbon_efficiency) + 20 * np.random.random()
            else:  # default
                carbon_efficiency = 0.5 + 0.25 * np.random.random()
                energy_consumption = 60 + 35 * (1 - carbon_efficiency) + 15 * np.random.random()
            
            # Performance metrics
            if scheduler == "performance-first":
                response_time = 50 + 30 * np.random.random()
                throughput = 800 + 200 * np.random.random()
            elif scheduler == "carbon-aware":
                response_time = 80 + 50 * np.random.random()
                throughput = 600 + 150 * np.random.random()
            else:
                response_time = 65 + 40 * np.random.random()
                throughput = 700 + 180 * np.random.random()
            
            # Resource utilization
            if node_type == "gpu-enabled":
                gpu_utilization = np.random.beta(3, 2) * 100
                energy_consumption *= 1.5  # GPUs consume more energy
            else:
                gpu_utilization = 0
            
            # Queue metrics
            queue_time = np.random.exponential(10)
            completion_rate = 0.85 + 0.14 * np.random.random()
            
            # Performance score (composite metric)
            performance_score = (
                0.3 * (100 - response_time) / 100 +
                0.3 * throughput / 1000 +
                0.2 * completion_rate +
                0.2 * (100 - queue_time) / 100
            )
            
            data.append({
                'experiment_id': f"exp_{i:04d}",
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 168)),
                'scheduler': scheduler,
                'node_type': node_type,
                'workload_type': workload_type,
                'carbon_efficiency': carbon_efficiency,
                'energy_consumption': energy_consumption,
                'performance_score': performance_score,
                'cpu_utilization': cpu_utilization,
                'memory_utilization': memory_utilization,
                'gpu_utilization': gpu_utilization,
                'response_time': response_time,
                'throughput': throughput,
                'queue_time': queue_time,
                'completion_rate': completion_rate,
                'resource_utilization': (cpu_utilization + memory_utilization) / 2
            })
        
        return pd.DataFrame(data)
    
    def generate_baseline_comparison_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate datasets for baseline comparison"""
        
        datasets = {}
        
        # Generate data for different baseline scenarios
        scenarios = {
            "kubernetes_default": {"n_samples": 500, "scheduler": "default"},
            "carbon_aware_v1": {"n_samples": 500, "scheduler": "carbon-aware"},
            "energy_efficient": {"n_samples": 500, "scheduler": "energy-efficient"},
            "performance_optimized": {"n_samples": 500, "scheduler": "performance-first"}
        }
        
        for scenario_name, config in scenarios.items():
            # Generate base dataset
            df = self.generate_carbon_efficiency_dataset(config["n_samples"])
            
            # Filter to specific scheduler if specified
            if "scheduler" in config:
                df = df[df['scheduler'] == config['scheduler']].copy()
            
            # Add scenario-specific noise and variations
            if scenario_name == "kubernetes_default":
                # Add more variability to default scheduler
                df['carbon_efficiency'] *= np.random.normal(1.0, 0.1, len(df))
                df['energy_consumption'] *= np.random.normal(1.0, 0.15, len(df))
            
            datasets[scenario_name] = df
        
        return datasets
    
    def generate_ablation_study_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate datasets for ablation studies"""
        
        datasets = {}
        
        # Base configuration
        base_config = {
            "carbon_awareness": True,
            "energy_optimization": True,
            "performance_tuning": True,
            "resource_balancing": True
        }
        
        # Generate ablation configurations
        ablation_configs = {}
        for feature in base_config.keys():
            config_name = f"without_{feature}"
            config = base_config.copy()
            config[feature] = False
            ablation_configs[config_name] = config
        
        # Add full configuration
        ablation_configs["full_config"] = base_config
        
        for config_name, config in ablation_configs.items():
            df = self.generate_carbon_efficiency_dataset(300)
            
            # Modify metrics based on disabled features
            if not config.get("carbon_awareness", True):
                df['carbon_efficiency'] *= 0.8  # Reduce carbon efficiency
                df['energy_consumption'] *= 1.2  # Increase energy consumption
            
            if not config.get("energy_optimization", True):
                df['energy_consumption'] *= 1.15
            
            if not config.get("performance_tuning", True):
                df['response_time'] *= 1.3
                df['throughput'] *= 0.85
                df['performance_score'] *= 0.9
            
            if not config.get("resource_balancing", True):
                df['resource_utilization'] *= np.random.normal(1.0, 0.2, len(df))
            
            datasets[config_name] = df
        
        return datasets
    
    def generate_time_series_dataset(self, days: int = 7) -> pd.DataFrame:
        """Generate time series data for temporal analysis"""
        
        start_time = datetime.now() - timedelta(days=days)
        end_time = datetime.now()
        
        # Generate timestamps (every 5 minutes)
        timestamps = pd.date_range(start_time, end_time, freq='5T')
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            # Add daily and weekly patterns
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Daily pattern (higher load during business hours)
            daily_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Weekly pattern (lower load on weekends)
            weekly_factor = 0.8 if day_of_week >= 5 else 1.0
            
            # Base load with patterns
            base_load = daily_factor * weekly_factor
            
            # Add noise
            noise = np.random.normal(0, 0.1)
            load_factor = max(0.1, base_load + noise)
            
            # Generate metrics based on load
            carbon_efficiency = 0.6 + 0.3 * (1 - load_factor) + 0.1 * np.random.random()
            energy_consumption = 40 + 60 * load_factor + 10 * np.random.random()
            performance_score = 0.7 + 0.2 * load_factor + 0.1 * np.random.random()
            
            data.append({
                'timestamp': timestamp,
                'load_factor': load_factor,
                'carbon_efficiency': carbon_efficiency,
                'energy_consumption': energy_consumption,
                'performance_score': performance_score,
                'cpu_utilization': 20 + 60 * load_factor + 10 * np.random.random(),
                'memory_utilization': 30 + 50 * load_factor + 15 * np.random.random(),
                'active_pods': int(10 + 50 * load_factor + 10 * np.random.random()),
                'scheduler': np.random.choice(['default', 'carbon-aware'], p=[0.3, 0.7])
            })
        
        return pd.DataFrame(data)
    
    def generate_benchmark_datasets(self):
        """Generate standard benchmark datasets"""
        
        benchmarks = {}
        
        # Small dataset for quick testing
        benchmarks['small'] = self.generate_carbon_efficiency_dataset(100)
        
        # Medium dataset for development
        benchmarks['medium'] = self.generate_carbon_efficiency_dataset(1000)
        
        # Large dataset for performance testing
        benchmarks['large'] = self.generate_carbon_efficiency_dataset(10000)
        
        # Time series benchmark
        benchmarks['timeseries'] = self.generate_time_series_dataset(30)
        
        return benchmarks
    
    def save_datasets(self):
        """Save all generated datasets"""
        
        print("Generating synthetic datasets...")
        
        # Main carbon efficiency dataset
        main_dataset = self.generate_carbon_efficiency_dataset(2000)
        main_dataset.to_csv(self.synthetic_path / "carbon_efficiency_main.csv", index=False)
        print(f"✓ Saved main dataset: {len(main_dataset)} samples")
        
        # Baseline comparison datasets
        baseline_datasets = self.generate_baseline_comparison_dataset()
        for name, df in baseline_datasets.items():
            df.to_csv(self.synthetic_path / f"baseline_{name}.csv", index=False)
            print(f"✓ Saved baseline dataset '{name}': {len(df)} samples")
        
        # Ablation study datasets
        ablation_datasets = self.generate_ablation_study_dataset()
        for name, df in ablation_datasets.items():
            df.to_csv(self.synthetic_path / f"ablation_{name}.csv", index=False)
            print(f"✓ Saved ablation dataset '{name}': {len(df)} samples")
        
        # Time series dataset
        timeseries_dataset = self.generate_time_series_dataset(14)
        timeseries_dataset.to_csv(self.synthetic_path / "timeseries_14days.csv", index=False)
        print(f"✓ Saved time series dataset: {len(timeseries_dataset)} samples")
        
        # Benchmark datasets
        benchmark_datasets = self.generate_benchmark_datasets()
        for name, df in benchmark_datasets.items():
            df.to_csv(self.benchmarks_path / f"benchmark_{name}.csv", index=False)
            print(f"✓ Saved benchmark dataset '{name}': {len(df)} samples")
        
        # Generate metadata
        self.generate_metadata()
        
        print("\n🎉 All datasets generated successfully!")
        print(f"📁 Synthetic datasets: {self.synthetic_path}")
        print(f"📁 Benchmark datasets: {self.benchmarks_path}")
    
    def generate_metadata(self):
        """Generate metadata for all datasets"""
        
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "generator_version": "1.0.0",
            "description": "Synthetic datasets for Carbon-Kube evaluation framework",
            "datasets": {
                "synthetic": {
                    "carbon_efficiency_main": {
                        "description": "Main carbon efficiency dataset with multiple schedulers",
                        "samples": 2000,
                        "features": [
                            "carbon_efficiency", "energy_consumption", "performance_score",
                            "cpu_utilization", "memory_utilization", "gpu_utilization",
                            "response_time", "throughput", "queue_time", "completion_rate"
                        ],
                        "schedulers": ["default", "carbon-aware", "energy-efficient", "performance-first"],
                        "use_case": "General evaluation and comparison"
                    },
                    "baseline_*": {
                        "description": "Datasets for baseline comparison studies",
                        "samples": 500,
                        "use_case": "Baseline performance evaluation"
                    },
                    "ablation_*": {
                        "description": "Datasets for ablation studies",
                        "samples": 300,
                        "use_case": "Feature importance analysis"
                    },
                    "timeseries_14days": {
                        "description": "Time series data with daily and weekly patterns",
                        "samples": "~4000 (5-minute intervals)",
                        "use_case": "Temporal analysis and trend detection"
                    }
                },
                "benchmarks": {
                    "benchmark_small": {
                        "description": "Small dataset for quick testing",
                        "samples": 100,
                        "use_case": "Unit tests and quick validation"
                    },
                    "benchmark_medium": {
                        "description": "Medium dataset for development",
                        "samples": 1000,
                        "use_case": "Development and integration testing"
                    },
                    "benchmark_large": {
                        "description": "Large dataset for performance testing",
                        "samples": 10000,
                        "use_case": "Performance and scalability testing"
                    },
                    "benchmark_timeseries": {
                        "description": "Time series benchmark dataset",
                        "samples": "~8640 (30 days, 5-minute intervals)",
                        "use_case": "Time series analysis benchmarking"
                    }
                }
            },
            "metrics": {
                "primary": ["carbon_efficiency", "energy_consumption", "performance_score"],
                "secondary": ["resource_utilization", "response_time", "throughput", "queue_time", "completion_rate"],
                "categorical": ["scheduler", "node_type", "workload_type"],
                "temporal": ["timestamp"]
            },
            "data_quality": {
                "missing_values": False,
                "outliers": "Realistic outliers included",
                "correlations": "Realistic correlations between metrics",
                "distributions": "Beta and normal distributions used for realism"
            }
        }
        
        # Save as YAML
        with open(self.base_path / "datasets_metadata.yaml", 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False, indent=2)
        
        # Save as JSON for programmatic access
        with open(self.base_path / "datasets_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print("✓ Generated dataset metadata")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for evaluation framework")
    parser.add_argument("--base-path", default="./evaluation/data", help="Base path for datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Generate datasets
    generator = DatasetGenerator(args.base_path)
    generator.save_datasets()

if __name__ == "__main__":
    main()