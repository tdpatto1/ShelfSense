# ShelfSense MVP Model Card

## Model Summary

ShelfSense uses two XGBoost regression pipelines:

- transaction model: predicts daily store-level transaction count
- sales model: predicts daily store and product-family unit sales

The sales model uses the transaction forecast as an input so demand forecasts can react to expected customer traffic.

## Intended Use

The model is intended for a course MVP demonstration of event-aware grocery forecasting. It helps a grocery manager estimate short-horizon demand and decide whether to increase, hold, or trim inventory orders.

## Not Intended For

This model should not be used for real purchasing decisions without retraining and validating on a real store's historical sales, inventory, and supplier lead-time data.

## Data

The training data is a reproducible calibrated synthetic dataset with:

- 4 Arizona store profiles
- 6 product families
- daily data from 2024-01-01 through 2025-12-31
- future forecast inputs for a 14-day planning horizon
- holidays, promotions, local event flags, payday flags, traffic proxies, and temperature signals

## Evaluation

The evaluation uses a time-based split with the final 56 days as the test period. Metrics include MAE, RMSE, WMAPE, event-day MAE, and non-event-day MAE. The MVP compares:

- seasonal naive 7-day lag baseline
- XGBoost without event features
- event-aware XGBoost

## Limitations

- The data is synthetic, not an actual retailer export.
- Local event labels are deterministic proxies, not live Ticketmaster/Nager.Date API pulls.
- Inventory recommendations use a simplified stock-on-hand assumption entered by the user.
- The model predicts product-family demand, not SKU-level demand.
- The explanation layer is operational, not causal.

## Next Model Improvements

- Add real Favorita or M5 data ingestion.
- Add SKU hierarchy and perishability constraints.
- Calibrate service levels using historical stockout/waste costs.
- Add prediction intervals instead of only point forecasts.
