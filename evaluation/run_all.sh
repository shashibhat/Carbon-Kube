#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-baseline}
DIR=$(cd "$(dirname "$0")" && pwd)
"$DIR/deploy_cluster.sh" "$MODE"
"$DIR/run_experiment.sh" "$MODE"
