#!/usr/bin/env bash
set -euo pipefail
python3 evaluation/figures/plot_scatter_pareto.py
python3 evaluation/figures/plot_timeseries_migrations.py
python3 evaluation/figures/plot_timeseries_migrations.py evaluation/figures/data/timeseries_exp1.csv timeseries_migrations_exp1.png
python3 evaluation/figures/plot_timeseries_migrations.py evaluation/figures/data/timeseries_exp2.csv timeseries_migrations_exp2.png
python3 evaluation/figures/plot_timeseries_migrations.py evaluation/figures/data/timeseries_exp3.csv timeseries_migrations_exp3.png
python3 evaluation/figures/plot_heatmap_scores.py
python3 evaluation/figures/plot_boxplots_metrics.py
python3 evaluation/figures/plot_rl_stacked_area.py
python3 evaluation/figures/plot_sankey_energy.py
echo "Figures written to evaluation/figures/out"
