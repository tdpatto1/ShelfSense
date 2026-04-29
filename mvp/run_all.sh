#!/usr/bin/env bash
set -euo pipefail

python src/generate_synthetic_data.py
python src/train_models.py
python src/make_forecast.py
python src/build_inventory_plan.py

echo "MVP artifacts created under mvp/artifacts"
echo "Launch demo with: python app.py"
