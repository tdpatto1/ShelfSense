# ShelfSense AI - Phase 3 MVP

ShelfSense is an event-aware grocery demand forecasting and inventory planning MVP. A store manager selects a store and product family, enters current stock assumptions, and receives:

- 7-day or 14-day sales forecast
- store traffic forecast
- event, promotion, holiday, and weather context
- recommended order quantity rounded to case-pack size
- downloadable daily inventory plan

## Folder Contents

- `app.py` - Gradio MVP dashboard
- `demo.py` - simple demo entrypoint that launches the same dashboard
- `src/generate_synthetic_data.py` - calibrated Arizona grocery scenario generator
- `src/train_models.py` - model training, baseline comparison, and evaluation
- `src/make_forecast.py` - recursive future sales and traffic forecasts
- `src/build_inventory_plan.py` - order recommendation tables
- `data/` - generated historical and future-input data
- `models/` - trained model checkpoints
- `artifacts/` - metrics, forecast outputs, plots, and inventory recommendations
- `report.md` - Phase 3 MVP report
- `MODEL_CARD.md` - model/data notes and limitations
- `requirements.txt` - Python dependencies
- `run_all.ps1` / `run_all.sh` - full reproduction scripts

## Quick Start On Windows

```powershell
cd C:\Users\gaelb\Downloads\shelfsenephase2\mvp
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
C:\Users\gaelb\AppData\Local\Programs\Python\Python312\python.exe app.py
```

Then open the local Gradio link shown in the terminal.

If dependencies need to be reinstalled:

```powershell
C:\Users\gaelb\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

## Manual Commands

```powershell
cd C:\Users\gaelb\Downloads\shelfsenephase2\mvp
C:\Users\gaelb\AppData\Local\Programs\Python\Python312\python.exe src/generate_synthetic_data.py
C:\Users\gaelb\AppData\Local\Programs\Python\Python312\python.exe src/train_models.py
C:\Users\gaelb\AppData\Local\Programs\Python\Python312\python.exe src/make_forecast.py
C:\Users\gaelb\AppData\Local\Programs\Python\Python312\python.exe src/build_inventory_plan.py
C:\Users\gaelb\AppData\Local\Programs\Python\Python312\python.exe app.py
```

You can also launch the same dashboard with:

```powershell
C:\Users\gaelb\AppData\Local\Programs\Python\Python312\python.exe demo.py
```

## Recreated Outputs

Running `run_all.ps1` recreates:

- `artifacts/metrics.csv`
- `artifacts/metrics_summary.json`
- `artifacts/future_sales_forecast.csv`
- `artifacts/future_transactions_forecast.csv`
- `artifacts/inventory_recommendations.csv`
- `artifacts/inventory_summary.csv`
- `artifacts/inventory_summary.md`
- model feature-importance files and plots

## Demo Flow

1. Choose a store.
2. Choose a product family.
3. Pick a 7-day or 14-day planning horizon.
4. Enter current stock on hand, service target, and case-pack size.
5. Review the recommended order quantity, forecast plots, daily plan, and downloadable order plan CSV.

## Data Note

The MVP uses a calibrated synthetic Arizona grocery scenario for reproducibility. It includes store-specific locations, local event calendars, promotions, holidays, payday effects, and weather proxies. The Phase 3 report explains why this was used and how it would be replaced with Favorita/M5 plus live holiday/event APIs in a production version.
