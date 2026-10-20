#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
$PY -m src.eval.parameter_count
