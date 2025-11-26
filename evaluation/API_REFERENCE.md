# Carbon-Kube Evaluation Framework API Reference

This document provides detailed API reference for the Carbon-Kube evaluation framework functions and utilities.

## Table of Contents

- [Data Loading and Processing](#data-loading-and-processing)
- [Statistical Analysis](#statistical-analysis)
- [Visualization](#visualization)
- [Baseline Comparison](#baseline-comparison)
- [Ablation Studies](#ablation-studies)
- [Utility Functions](#utility-functions)

## Data Loading and Processing

### `load_dataset(file_path, **kwargs)`

Load and validate a dataset for evaluation.

**Parameters:**
- `file_path` (str): Path to the CSV file
- `**kwargs`: Additional arguments passed to `pd.read_csv()`

**Returns:**
- `pd.DataFrame`: Loaded and validated dataset

**Example:**
```python
data = load_dataset('data/synthetic/main_dataset.csv')
```

### `validate_dataset(data, required_columns=None)`

Validate dataset structure and content.

**Parameters:**
- `data` (pd.DataFrame): Dataset to validate
- `required_columns` (list, optional): List of required column names

**Returns:**
- `bool`: True if validation passes

**Raises:**
- `ValueError`: If validation fails

**Example:**
```python
is_valid = validate_dataset(data, ['scheduler', 'carbon_efficiency'])
```

### `preprocess_data(data, normalize=False, handle_outliers=True)`

Preprocess data for analysis.

**Parameters:**
- `data` (pd.DataFrame): Raw dataset
- `normalize` (bool): Whether to normalize numeric columns
- `handle_outliers` (bool): Whether to handle outliers

**Returns:**
- `pd.DataFrame`: Preprocessed dataset

**Example:**
```python
clean_data = preprocess_data(data, normalize=True)
```

## Statistical Analysis

### `perform_ttest(group1, group2, alternative='two-sided')`

Perform independent t-test between two groups.

**Parameters:**
- `group1` (array-like): First group data
- `group2` (array-like): Second group data
- `alternative` (str): Alternative hypothesis ('two-sided', 'less', 'greater')

**Returns:**
- `dict`: Dictionary containing test statistic, p-value, and interpretation

**Example:**
```python
result = perform_ttest(
    data[data['scheduler'] == 'default']['carbon_efficiency'],
    data[data['scheduler'] == 'carbon_aware']['carbon_efficiency']
)
```

### `calculate_effect_size(group1, group2, method='cohen_d')`

Calculate effect size between two groups.

**Parameters:**
- `group1` (array-like): First group data
- `group2` (array-like): Second group data
- `method` (str): Effect size method ('cohen_d', 'glass_delta', 'hedges_g')

**Returns:**
- `float`: Effect size value

**Example:**
```python
effect_size = calculate_effect_size(group1, group2, method='cohen_d')
```

### `bootstrap_confidence_interval(data, statistic=np.mean, n_bootstrap=1000, confidence_level=0.95)`

Calculate bootstrap confidence interval for a statistic.

**Parameters:**
- `data` (array-like): Input data
- `statistic` (callable): Function to calculate statistic
- `n_bootstrap` (int): Number of bootstrap samples
- `confidence_level` (float): Confidence level (0-1)

**Returns:**
- `tuple`: Lower and upper confidence interval bounds

**Example:**
```python
ci_lower, ci_upper = bootstrap_confidence_interval(
    data['carbon_efficiency'], 
    statistic=np.mean,
    n_bootstrap=1000
)
```

### `perform_anova(data, groups, dependent_var)`

Perform one-way ANOVA analysis.

**Parameters:**
- `data` (pd.DataFrame): Dataset
- `groups` (str): Column name for grouping variable
- `dependent_var` (str): Column name for dependent variable

**Returns:**
- `dict`: ANOVA results including F-statistic, p-value, and post-hoc tests

**Example:**
```python
anova_results = perform_anova(data, 'scheduler', 'carbon_efficiency')
```

### `multiple_comparison_correction(p_values, method='bonferroni')`

Apply multiple comparison correction to p-values.

**Parameters:**
- `p_values` (array-like): List of p-values
- `method` (str): Correction method ('bonferroni', 'fdr_bh', 'holm')

**Returns:**
- `tuple`: Corrected p-values and rejection decisions

**Example:**
```python
corrected_p, rejected = multiple_comparison_correction(
    [0.01, 0.03, 0.08], 
    method='bonferroni'
)
```

## Visualization

### `create_comparison_plot(data, x_col, y_col, hue_col=None, plot_type='box')`

Create comparison plots for different schedulers or conditions.

**Parameters:**
- `data` (pd.DataFrame): Dataset
- `x_col` (str): Column for x-axis
- `y_col` (str): Column for y-axis
- `hue_col` (str, optional): Column for color grouping
- `plot_type` (str): Type of plot ('box', 'violin', 'bar', 'scatter')

**Returns:**
- `matplotlib.figure.Figure`: Plot figure

**Example:**
```python
fig = create_comparison_plot(
    data, 
    x_col='scheduler', 
    y_col='carbon_efficiency',
    plot_type='box'
)
```

### `create_correlation_heatmap(data, columns=None, method='pearson')`

Create correlation heatmap for numeric variables.

**Parameters:**
- `data` (pd.DataFrame): Dataset
- `columns` (list, optional): Specific columns to include
- `method` (str): Correlation method ('pearson', 'spearman', 'kendall')

**Returns:**
- `matplotlib.figure.Figure`: Heatmap figure

**Example:**
```python
fig = create_correlation_heatmap(
    data, 
    columns=['carbon_efficiency', 'energy_consumption', 'performance_score']
)
```

### `create_time_series_plot(data, time_col, value_col, group_col=None)`

Create time series visualization.

**Parameters:**
- `data` (pd.DataFrame): Dataset with time series data
- `time_col` (str): Column containing timestamps
- `value_col` (str): Column containing values to plot
- `group_col` (str, optional): Column for grouping lines

**Returns:**
- `matplotlib.figure.Figure`: Time series plot

**Example:**
```python
fig = create_time_series_plot(
    data, 
    time_col='timestamp', 
    value_col='carbon_efficiency',
    group_col='scheduler'
)
```

### `create_distribution_plot(data, column, by_group=None, plot_type='hist')`

Create distribution plots for variables.

**Parameters:**
- `data` (pd.DataFrame): Dataset
- `column` (str): Column to plot distribution for
- `by_group` (str, optional): Column to group distributions by
- `plot_type` (str): Type of plot ('hist', 'kde', 'both')

**Returns:**
- `matplotlib.figure.Figure`: Distribution plot

**Example:**
```python
fig = create_distribution_plot(
    data, 
    column='carbon_efficiency',
    by_group='scheduler',
    plot_type='both'
)
```

## Baseline Comparison

### `compare_baselines(data, baseline_col, metrics, reference_baseline=None)`

Compare multiple baselines across specified metrics.

**Parameters:**
- `data` (pd.DataFrame): Dataset containing baseline results
- `baseline_col` (str): Column containing baseline names
- `metrics` (list): List of metric columns to compare
- `reference_baseline` (str, optional): Reference baseline for relative comparisons

**Returns:**
- `dict`: Comprehensive comparison results

**Example:**
```python
comparison = compare_baselines(
    data,
    baseline_col='scheduler',
    metrics=['carbon_efficiency', 'energy_consumption', 'performance_score'],
    reference_baseline='kubernetes_default'
)
```

### `rank_baselines(data, baseline_col, metrics, weights=None)`

Rank baselines based on multiple metrics.

**Parameters:**
- `data` (pd.DataFrame): Dataset
- `baseline_col` (str): Column containing baseline names
- `metrics` (list): List of metric columns
- `weights` (dict, optional): Weights for each metric

**Returns:**
- `pd.DataFrame`: Ranked baselines with scores

**Example:**
```python
rankings = rank_baselines(
    data,
    baseline_col='scheduler',
    metrics=['carbon_efficiency', 'energy_consumption'],
    weights={'carbon_efficiency': 0.7, 'energy_consumption': 0.3}
)
```

### `calculate_improvement_percentage(baseline_data, reference_data, metric)`

Calculate percentage improvement over reference.

**Parameters:**
- `baseline_data` (array-like): Baseline metric values
- `reference_data` (array-like): Reference metric values
- `metric` (str): Metric name for interpretation

**Returns:**
- `float`: Percentage improvement (positive = better)

**Example:**
```python
improvement = calculate_improvement_percentage(
    carbon_aware_data['carbon_efficiency'],
    default_data['carbon_efficiency'],
    'carbon_efficiency'
)
```

## Ablation Studies

### `perform_ablation_study(data, feature_cols, target_col, baseline_model=None)`

Perform ablation study to understand feature importance.

**Parameters:**
- `data` (pd.DataFrame): Dataset
- `feature_cols` (list): List of feature columns
- `target_col` (str): Target variable column
- `baseline_model` (sklearn estimator, optional): Baseline model to use

**Returns:**
- `dict`: Ablation study results with feature importance rankings

**Example:**
```python
ablation_results = perform_ablation_study(
    data,
    feature_cols=['carbon_awareness', 'energy_optimization', 'load_balancing'],
    target_col='carbon_efficiency'
)
```

### `analyze_feature_interactions(data, feature_cols, target_col, interaction_depth=2)`

Analyze interactions between features in ablation study.

**Parameters:**
- `data` (pd.DataFrame): Dataset
- `feature_cols` (list): List of feature columns
- `target_col` (str): Target variable column
- `interaction_depth` (int): Depth of interactions to analyze

**Returns:**
- `dict`: Interaction analysis results

**Example:**
```python
interactions = analyze_feature_interactions(
    data,
    feature_cols=['carbon_awareness', 'energy_optimization'],
    target_col='carbon_efficiency',
    interaction_depth=2
)
```

## Utility Functions

### `load_config(config_path='config/evaluation_config.yaml')`

Load configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `dict`: Configuration dictionary

**Example:**
```python
config = load_config('config/evaluation_config.yaml')
```

### `save_results(results, output_path, format='json')`

Save analysis results to file.

**Parameters:**
- `results` (dict): Results dictionary
- `output_path` (str): Output file path
- `format` (str): Output format ('json', 'csv', 'pickle')

**Returns:**
- `bool`: True if save successful

**Example:**
```python
success = save_results(
    comparison_results,
    'results/baseline_comparison.json',
    format='json'
)
```

### `generate_report(results, template='default', output_format='html')`

Generate analysis report from results.

**Parameters:**
- `results` (dict): Analysis results
- `template` (str): Report template to use
- `output_format` (str): Output format ('html', 'pdf', 'markdown')

**Returns:**
- `str`: Path to generated report

**Example:**
```python
report_path = generate_report(
    statistical_results,
    template='statistical_analysis',
    output_format='html'
)
```

### `check_assumptions(data, test_type='normality')`

Check statistical assumptions for analysis.

**Parameters:**
- `data` (array-like): Data to test
- `test_type` (str): Type of assumption test ('normality', 'homoscedasticity', 'independence')

**Returns:**
- `dict`: Test results and recommendations

**Example:**
```python
assumption_results = check_assumptions(
    data['carbon_efficiency'],
    test_type='normality'
)
```

### `calculate_sample_size(effect_size, power=0.8, alpha=0.05, test_type='ttest')`

Calculate required sample size for statistical power.

**Parameters:**
- `effect_size` (float): Expected effect size
- `power` (float): Desired statistical power
- `alpha` (float): Significance level
- `test_type` (str): Type of statistical test

**Returns:**
- `int`: Required sample size per group

**Example:**
```python
n_required = calculate_sample_size(
    effect_size=0.5,
    power=0.8,
    alpha=0.05,
    test_type='ttest'
)
```

## Error Handling

### Common Exceptions

- `DataValidationError`: Raised when dataset validation fails
- `InsufficientDataError`: Raised when sample size is too small
- `ConfigurationError`: Raised when configuration is invalid
- `AnalysisError`: Raised when statistical analysis fails

### Example Error Handling

```python
try:
    results = perform_ttest(group1, group2)
except InsufficientDataError as e:
    print(f"Sample size too small: {e}")
    # Handle small sample case
except DataValidationError as e:
    print(f"Data validation failed: {e}")
    # Handle data issues
```

## Configuration

### Default Configuration Structure

```yaml
analysis:
  significance_level: 0.05
  confidence_level: 0.95
  bootstrap_iterations: 1000
  multiple_comparison_method: "bonferroni"

visualization:
  figure_size: [10, 6]
  color_palette: "Set2"
  save_format: "png"
  dpi: 300

data:
  required_columns:
    - scheduler
    - carbon_efficiency
    - energy_consumption
  optional_columns:
    - performance_score
    - response_time
    - throughput

output:
  results_directory: "results"
  report_directory: "reports"
  figure_directory: "figures"
```

## Best Practices

### Data Preparation
1. Always validate data before analysis
2. Handle missing values appropriately
3. Check for outliers and decide on treatment
4. Ensure sufficient sample sizes

### Statistical Analysis
1. Check assumptions before applying tests
2. Use appropriate tests for data distribution
3. Apply multiple comparison corrections
4. Report effect sizes alongside p-values

### Visualization
1. Choose appropriate plot types for data
2. Include error bars or confidence intervals
3. Use consistent color schemes
4. Provide clear labels and legends

### Reproducibility
1. Set random seeds for reproducible results
2. Document all analysis parameters
3. Save intermediate results
4. Version control analysis scripts

## Examples

### Complete Analysis Workflow

```python
# Load and validate data
data = load_dataset('data/baseline_comparison.csv')
validate_dataset(data, ['scheduler', 'carbon_efficiency'])

# Preprocess data
clean_data = preprocess_data(data, normalize=False, handle_outliers=True)

# Perform baseline comparison
comparison_results = compare_baselines(
    clean_data,
    baseline_col='scheduler',
    metrics=['carbon_efficiency', 'energy_consumption', 'performance_score']
)

# Statistical testing
for scheduler in clean_data['scheduler'].unique():
    if scheduler != 'kubernetes_default':
        group1 = clean_data[clean_data['scheduler'] == 'kubernetes_default']['carbon_efficiency']
        group2 = clean_data[clean_data['scheduler'] == scheduler]['carbon_efficiency']
        
        test_result = perform_ttest(group1, group2)
        effect_size = calculate_effect_size(group1, group2)
        
        print(f"{scheduler} vs default: p={test_result['p_value']:.3f}, d={effect_size:.3f}")

# Create visualizations
fig1 = create_comparison_plot(clean_data, 'scheduler', 'carbon_efficiency', plot_type='box')
fig2 = create_correlation_heatmap(clean_data, ['carbon_efficiency', 'energy_consumption', 'performance_score'])

# Save results
save_results(comparison_results, 'results/baseline_comparison.json')
```

This API reference provides comprehensive documentation for using the Carbon-Kube evaluation framework programmatically. For interactive usage examples, refer to the Jupyter notebooks in the `notebooks/` directory.