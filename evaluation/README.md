# Carbon-Kube Evaluation Framework

A comprehensive evaluation framework for carbon-efficient Kubernetes scheduling algorithms, providing rigorous statistical analysis, baseline comparisons, and reproducible evaluation methodologies.

## 🎯 Overview

The Carbon-Kube evaluation framework is designed to help researchers and practitioners evaluate, compare, and validate carbon-efficient Kubernetes scheduling algorithms. It provides a complete toolkit for performance analysis, statistical validation, and reproducible research.

### Key Features

- **Multi-Metric Evaluation**: Carbon efficiency, energy consumption, performance metrics
- **Statistical Analysis**: Hypothesis testing, effect size analysis, confidence intervals
- **Baseline Comparisons**: Compare against standard Kubernetes schedulers
- **Ablation Studies**: Understand component contributions and feature importance
- **Reproducible Research**: Consistent evaluation processes and artifact management
- **Interactive Analysis**: Jupyter notebooks for exploratory data analysis

## 📊 Supported Metrics

### Primary Metrics
- **Carbon Efficiency**: Environmental impact measurement
- **Energy Consumption**: Power usage optimization
- **Performance Score**: Overall system performance

### Secondary Metrics
- **Response Time**: Request latency measurements
- **Throughput**: Requests per second capacity
- **Resource Utilization**: CPU, memory, network usage
- **System Reliability**: Availability and error rates

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)
- Kubernetes cluster data in CSV format

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Carbon-Kube-1/evaluation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # or run the setup script
   ./setup.sh
   ```

3. **Generate example datasets**:
   ```bash
   cd data
   python generate_datasets.py
   ```

4. **Start Jupyter**:
   ```bash
   jupyter notebook notebooks/
   ```

### Basic Usage

1. **Start with the overview**: Open `00_Framework_Overview.ipynb`
2. **Follow the getting started guide**: Run `01_Getting_Started.ipynb`
3. **Perform baseline comparisons**: Use `03_Baseline_Comparison.ipynb`
4. **Conduct statistical analysis**: Execute `04_Statistical_Analysis.ipynb`

## 📁 Directory Structure

```
evaluation/
├── data/                           # Dataset storage
│   ├── raw/                       # Raw data files
│   ├── synthetic/                 # Generated synthetic datasets
│   ├── benchmarks/                # Benchmark datasets
│   ├── dataset_config.yaml        # Dataset configuration
│   └── generate_datasets.py       # Dataset generation script
├── notebooks/                      # Analysis notebooks
│   ├── 00_Framework_Overview.ipynb    # Framework introduction
│   ├── 01_Getting_Started.ipynb       # Basic usage guide
│   ├── 02_Ablation_Studies.ipynb      # Feature importance analysis
│   ├── 03_Baseline_Comparison.ipynb   # Scheduler comparisons
│   └── 04_Statistical_Analysis.ipynb  # Advanced statistics
├── results/                        # Analysis outputs
├── scripts/                        # Utility scripts
├── config/                         # Configuration files
├── setup.py                        # Package setup
├── setup.sh                        # Environment setup script
└── README.md                       # This file
```

## 📚 Notebook Guide

### 1. Framework Overview (`00_Framework_Overview.ipynb`)
- **Purpose**: Introduction to the evaluation framework
- **Content**: Architecture, features, best practices
- **Audience**: All users, especially newcomers

### 2. Getting Started (`01_Getting_Started.ipynb`)
- **Purpose**: Basic framework usage and data exploration
- **Content**: Data loading, visualization, basic statistics
- **Audience**: New users, initial analysis

### 3. Ablation Studies (`02_Ablation_Studies.ipynb`)
- **Purpose**: Understanding component contributions
- **Content**: Feature importance, interaction effects, ML-based analysis
- **Audience**: Algorithm developers, researchers

### 4. Baseline Comparison (`03_Baseline_Comparison.ipynb`)
- **Purpose**: Comprehensive scheduler comparison
- **Content**: Multi-scheduler analysis, trade-off studies, recommendations
- **Audience**: Performance evaluation, scheduler selection

### 5. Statistical Analysis (`04_Statistical_Analysis.ipynb`)
- **Purpose**: Advanced statistical methods
- **Content**: Hypothesis testing, bootstrap methods, effect sizes
- **Audience**: Researchers, publication-quality analysis

## 📊 Data Format

### Required Columns

Your dataset must include these essential columns:

```csv
scheduler,carbon_efficiency,energy_consumption,performance_score
kubernetes_default,0.75,120.5,0.82
carbon_aware_v1,0.89,98.2,0.85
energy_efficient,0.82,105.1,0.88
```

### Optional Columns

Additional columns for enhanced analysis:

- `response_time`: Request response time (ms)
- `throughput`: Requests per second
- `cpu_utilization`: CPU usage percentage (0-100)
- `memory_utilization`: Memory usage percentage (0-100)
- `workload_type`: Type of workload (web_service, batch_job, etc.)
- `node_type`: Node configuration (small, medium, large)
- `timestamp`: Time of measurement (ISO format)

### Data Quality Requirements

- **No missing values** in required columns
- **Consistent units** across measurements
- **Sufficient sample sizes** (minimum 30 observations per scheduler)
- **Balanced datasets** when comparing schedulers

## ⚙️ Configuration

### Dataset Configuration (`data/dataset_config.yaml`)

```yaml
datasets:
  main_dataset:
    description: "Primary evaluation dataset"
    file: "main_dataset.csv"
    metrics:
      - carbon_efficiency
      - energy_consumption
      - performance_score
    
  baseline_comparison:
    schedulers:
      - kubernetes_default
      - carbon_aware_v1
      - energy_efficient
      - hybrid_balanced
    
analysis_settings:
  significance_level: 0.05
  bootstrap_iterations: 1000
  confidence_level: 0.95
  multiple_comparison_correction: "bonferroni"
```

### Environment Variables

```bash
# Optional: Set custom paths
export CARBON_KUBE_DATA_PATH="/path/to/evaluation/data"
export CARBON_KUBE_RESULTS_PATH="/path/to/evaluation/results"
```

## 🔬 Analysis Methods

### Statistical Tests
- **Parametric Tests**: t-tests, ANOVA, linear regression
- **Non-parametric Tests**: Mann-Whitney U, Kruskal-Wallis
- **Effect Size Analysis**: Cohen's d, eta-squared
- **Multiple Comparisons**: Bonferroni, FDR corrections

### Bootstrap Methods
- **Confidence Intervals**: Robust estimation without distributional assumptions
- **Hypothesis Testing**: Bootstrap-based significance testing
- **Resampling**: Monte Carlo methods for uncertainty quantification

### Machine Learning
- **Feature Importance**: Random Forest, permutation importance
- **Dimensionality Reduction**: PCA, t-SNE
- **Clustering**: K-means, hierarchical clustering

## 📈 Example Workflows

### 1. Basic Scheduler Comparison

```python
# Load data
import pandas as pd
data = pd.read_csv('data/synthetic/baseline_comparison.csv')

# Basic comparison
schedulers = data['scheduler'].unique()
for scheduler in schedulers:
    subset = data[data['scheduler'] == scheduler]
    print(f"{scheduler}: {subset['carbon_efficiency'].mean():.3f}")
```

### 2. Statistical Significance Testing

```python
from scipy.stats import ttest_ind

# Compare two schedulers
group1 = data[data['scheduler'] == 'kubernetes_default']['carbon_efficiency']
group2 = data[data['scheduler'] == 'carbon_aware_v1']['carbon_efficiency']

t_stat, p_value = ttest_ind(group1, group2)
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
```

### 3. Bootstrap Confidence Intervals

```python
import numpy as np

def bootstrap_mean(data, n_bootstrap=1000):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    return np.percentile(bootstrap_means, [2.5, 97.5])

ci = bootstrap_mean(group2)
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## 🎯 Best Practices

### Statistical Analysis
- **Check assumptions** before applying parametric tests
- **Use appropriate tests** based on data distribution
- **Report effect sizes** alongside p-values
- **Apply multiple comparison corrections** when testing multiple hypotheses
- **Provide confidence intervals** for all estimates

### Data Quality
- **Validate data** before analysis (missing values, outliers)
- **Document data sources** and collection methods
- **Use version control** for datasets
- **Ensure reproducibility** with random seeds

### Visualization
- **Choose appropriate charts** for data types
- **Include error bars** or confidence intervals
- **Use consistent color schemes** across analyses
- **Provide clear labels** and legends

### Reporting
- **Define all metrics** and their units clearly
- **Report statistical details** (test statistics, p-values, effect sizes)
- **Acknowledge limitations** and assumptions
- **Provide actionable recommendations**

## 🔧 Troubleshooting

### Common Issues

#### Data Loading Problems
```python
# Check file paths
import os
print(os.getcwd())
print(os.listdir('data/'))

# Handle encoding issues
data = pd.read_csv('file.csv', encoding='utf-8')
```

#### Statistical Analysis Issues
```python
# Check for normality
from scipy.stats import shapiro
stat, p = shapiro(data['carbon_efficiency'])
print(f"Normality test p-value: {p:.3f}")

# Handle small sample sizes
if len(data) < 30:
    print("Consider using non-parametric tests")
```

#### Memory Issues
```python
# Use chunked processing for large datasets
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    pass
```

### Getting Help

1. **Check documentation** in notebooks and comments
2. **Review error messages** carefully
3. **Consult statistical references** for method details
4. **Use built-in help**: `help(function_name)` or `?function_name`

## 📚 References

### Statistical Methods
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate

### Carbon-Efficient Computing
- Beloglazov, A., & Buyya, R. (2012). Optimal online deterministic algorithms and adaptive heuristics for energy and performance efficient dynamic consolidation of virtual machines in cloud data centers
- Koomey, J., et al. (2011). Implications of historical trends in the electrical efficiency of computing

### Tools and Libraries
- **pandas**: McKinney, W. (2010). Data structures for statistical computing in Python
- **scipy**: Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing
- **scikit-learn**: Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python

## 🤝 Contributing

We welcome contributions to improve the evaluation framework:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-analysis`
3. **Make your changes** and add tests
4. **Submit a pull request** with detailed description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest test/

# Check code style
flake8 evaluation/
black evaluation/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- The Kubernetes community for scheduling framework insights
- The scientific Python ecosystem for statistical computing tools
- Carbon-aware computing research community for domain expertise

---

**Happy analyzing! 🎉**

For questions, issues, or contributions, please visit our [GitHub repository](https://github.com/your-org/carbon-kube) or contact the maintainers.