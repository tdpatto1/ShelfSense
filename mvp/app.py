from __future__ import annotations

import math
import os
from pathlib import Path

import gradio as gr
import pandas as pd


ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))

import matplotlib.pyplot as plt


ARTIFACTS = ROOT / 'artifacts'

forecast_df = pd.read_csv(ARTIFACTS / 'future_sales_forecast.csv', parse_dates=['date'])
transaction_df = pd.read_csv(ARTIFACTS / 'future_transactions_forecast.csv', parse_dates=['date'])

stores = sorted(forecast_df['store_id'].unique())
families = sorted(forecast_df['family'].unique())

CASE_PACKS = {
    'produce': 12,
    'beverages': 24,
    'snacks': 18,
    'dairy': 12,
    'bakery': 8,
    'frozen': 10,
}

SERVICE_LEVELS = {
    'Lean': 0.06,
    'Balanced': 0.14,
    'High availability': 0.24,
}


def pct_text(value: float) -> str:
    sign = '+' if value >= 0 else ''
    return f'{sign}{value:.1f}%'


def round_to_case_pack(units: float, case_pack: int) -> int:
    if units <= 0:
        return 0
    return int(math.ceil(units / case_pack) * case_pack)


def select_data(store_id: str, family: str, horizon_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = (
        forecast_df[(forecast_df['store_id'] == store_id) & (forecast_df['family'] == family)]
        .copy()
        .sort_values('date')
        .head(int(horizon_days))
    )
    traffic = (
        transaction_df[transaction_df['store_id'] == store_id]
        .copy()
        .sort_values('date')
        .head(int(horizon_days))
    )
    return sales, traffic


def build_recommendation(
    store_id: str,
    family: str,
    horizon_days: int,
    current_stock: float,
    service_level: str,
    case_pack: int,
) -> dict[str, float | int | str]:
    sales, traffic = select_data(store_id, family, horizon_days)
    total_forecast = float(sales['pred_sales'].sum())
    total_baseline = float(sales['baseline_sales'].sum())
    uplift_pct = ((total_forecast - total_baseline) / total_baseline) * 100 if total_baseline else 0
    safety_stock = total_forecast * SERVICE_LEVELS[service_level]
    current_stock = 0 if current_stock is None else float(current_stock)
    net_need = total_forecast + safety_stock - current_stock
    recommended_order = round_to_case_pack(net_need, int(case_pack))
    peak_sales = sales.sort_values('pred_sales', ascending=False).iloc[0]
    peak_traffic = traffic.sort_values('pred_transactions', ascending=False).iloc[0]
    event_days = int(((sales['holiday_flag'] == 1) | (sales['local_event_flag'] == 1)).sum())
    promo_days = int(sales['onpromotion'].sum())

    if recommended_order > 0 and (uplift_pct >= 15 or event_days):
        priority = 'High'
    elif recommended_order > 0 or uplift_pct >= 8:
        priority = 'Medium'
    else:
        priority = 'Normal'

    if recommended_order > 0:
        action = f'Order {recommended_order:,} units'
    elif uplift_pct < -8:
        action = 'Hold or reduce order'
    else:
        action = 'Hold current order'

    return {
        'total_forecast': total_forecast,
        'total_baseline': total_baseline,
        'uplift_pct': uplift_pct,
        'safety_stock': safety_stock,
        'recommended_order': recommended_order,
        'peak_sales_date': peak_sales['date'].strftime('%b %d'),
        'peak_sales_units': float(peak_sales['pred_sales']),
        'peak_traffic_date': peak_traffic['date'].strftime('%b %d'),
        'peak_transactions': float(peak_traffic['pred_transactions']),
        'event_days': event_days,
        'promo_days': promo_days,
        'priority': priority,
        'action': action,
    }


def build_kpis(rec: dict[str, float | int | str]) -> str:
    cards = [
        ('Forecast Demand', f"{rec['total_forecast']:,.0f}", 'Units across selected horizon'),
        ('Demand Lift', pct_text(float(rec['uplift_pct'])), 'Compared with weekday baseline'),
        ('Recommended Order', f"{rec['recommended_order']:,.0f}", str(rec['action'])),
        ('Priority', str(rec['priority']), f"{rec['event_days']} event days, {rec['promo_days']} promo days"),
    ]
    html = ['<div class="metrics-grid">']
    for title, value, note in cards:
        html.append(
            f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """
        )
    html.append('</div>')
    return ''.join(html)


def build_brief(store_id: str, family: str, horizon_days: int, rec: dict[str, float | int | str], sales: pd.DataFrame) -> str:
    active_drivers = sorted(
        {
            value.replace('_', ' ')
            for value in pd.concat([sales['holiday_name'], sales['local_event_name']]).unique()
            if value != 'none'
        }
    )
    driver_text = ', '.join(active_drivers) if active_drivers else 'weekday pattern and recent sales history'
    profile = sales.iloc[0]['store_profile'].title()
    city = sales.iloc[0]['city']
    weather_band = f"{sales['avg_temp_f'].min():.0f}F to {sales['avg_temp_f'].max():.0f}F"

    return (
        f"## {store_id} - {family.title()} Inventory Plan\n"
        f"**Store:** {profile} in {city}\n\n"
        f"For the next **{horizon_days} days**, ShelfSense forecasts **{rec['total_forecast']:,.0f} units** "
        f"versus a baseline of **{rec['total_baseline']:,.0f} units**. The recommended action is "
        f"**{rec['action']}** with a **{rec['priority']}** priority level.\n\n"
        f"**Peak sales day:** {rec['peak_sales_date']} at {rec['peak_sales_units']:.1f} units. "
        f"**Peak traffic day:** {rec['peak_traffic_date']} at {rec['peak_transactions']:.0f} transactions.\n\n"
        f"**Likely drivers:** {driver_text}. **Weather range:** {weather_band}."
    )


def build_order_table(
    sales: pd.DataFrame,
    current_stock: float,
    service_level: str,
    case_pack: int,
) -> pd.DataFrame:
    running_stock = 0 if current_stock is None else float(current_stock)
    rows = []
    for _, row in sales.iterrows():
        safety_units = row['pred_sales'] * SERVICE_LEVELS[service_level]
        net_need = row['pred_sales'] + safety_units - running_stock
        order_units = round_to_case_pack(net_need, int(case_pack))
        running_stock = max(0, running_stock + order_units - row['pred_sales'])
        delta = row['pred_sales'] - row['baseline_sales']
        uplift = (delta / row['baseline_sales']) * 100 if row['baseline_sales'] else 0
        drivers = []
        if row['holiday_name'] != 'none':
            drivers.append(row['holiday_name'].replace('_', ' '))
        if row['local_event_name'] != 'none':
            drivers.append(row['local_event_name'].replace('_', ' '))
        if row['onpromotion']:
            drivers.append('promotion')
        if row['heatwave_flag']:
            drivers.append('heat')
        rows.append(
            {
                'Date': row['date'].strftime('%Y-%m-%d'),
                'Forecast Units': round(row['pred_sales'], 1),
                'Baseline Units': round(row['baseline_sales'], 1),
                'Uplift %': round(uplift, 1),
                'Order Units': order_units,
                'Projected Ending Stock': round(running_stock, 1),
                'Drivers': ', '.join(drivers) if drivers else 'baseline demand',
            }
        )
    return pd.DataFrame(rows)


def build_forecast_table(sales: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    merged = sales.merge(
        traffic[['date', 'pred_transactions']],
        on='date',
        how='left',
        suffixes=('', '_store'),
    )
    out = merged[
        ['date', 'pred_sales', 'baseline_sales', 'pred_transactions', 'onpromotion', 'holiday_name', 'local_event_name', 'explanation']
    ].copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    return out.rename(
        columns={
            'date': 'Date',
            'pred_sales': 'Forecast Units',
            'baseline_sales': 'Baseline Units',
            'pred_transactions': 'Traffic Forecast',
            'onpromotion': 'Promotion',
            'holiday_name': 'Holiday',
            'local_event_name': 'Local Event',
            'explanation': 'Manager Explanation',
        }
    ).round({'Forecast Units': 1, 'Baseline Units': 1, 'Traffic Forecast': 0})


def make_sales_plot(sales: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    ax.plot(sales['date'], sales['pred_sales'], color='#0f766e', linewidth=2.6, label='Forecast')
    ax.plot(sales['date'], sales['baseline_sales'], color='#a16207', linewidth=2, linestyle='--', label='Baseline')
    promo = sales[sales['onpromotion'] == 1]
    if not promo.empty:
        ax.scatter(promo['date'], promo['pred_sales'], color='#dc2626', s=42, label='Promotion')
    for _, row in sales[(sales['holiday_flag'] == 1) | (sales['local_event_flag'] == 1)].iterrows():
        ax.axvspan(row['date'] - pd.Timedelta(hours=12), row['date'] + pd.Timedelta(hours=12), color='#fde68a', alpha=0.32)
    ax.set_title('Demand Forecast', loc='left', fontweight='bold')
    ax.set_ylabel('Units')
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc='upper left')
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def make_traffic_plot(traffic: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    ax.plot(traffic['date'], traffic['pred_transactions'], color='#1d4ed8', linewidth=2.6)
    peak = traffic.sort_values('pred_transactions', ascending=False).iloc[0]
    ax.scatter([peak['date']], [peak['pred_transactions']], color='#b91c1c', s=58)
    ax.set_title('Store Traffic Forecast', loc='left', fontweight='bold')
    ax.set_ylabel('Transactions')
    ax.grid(alpha=0.18)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def export_selected_plan(order_table: pd.DataFrame, store_id: str, family: str) -> str:
    out_path = ARTIFACTS / f'selected_order_plan_{store_id}_{family}.csv'
    order_table.to_csv(out_path, index=False)
    return str(out_path)


def load_view(
    store_id: str,
    family: str,
    horizon_days: int,
    current_stock: float,
    service_level: str,
    case_pack: int,
):
    sales, traffic = select_data(store_id, family, int(horizon_days))
    rec = build_recommendation(store_id, family, int(horizon_days), current_stock, service_level, int(case_pack))
    brief = build_brief(store_id, family, int(horizon_days), rec, sales)
    kpis = build_kpis(rec)
    order_table = build_order_table(sales, current_stock, service_level, int(case_pack))
    forecast_table = build_forecast_table(sales, traffic)
    sales_plot = make_sales_plot(sales)
    traffic_plot = make_traffic_plot(traffic)
    export_path = export_selected_plan(order_table, store_id, family)
    return brief, kpis, sales_plot, traffic_plot, order_table, forecast_table, export_path


def default_case_pack(family: str) -> gr.Number:
    return CASE_PACKS.get(family, 12)


CSS = """
.gradio-container {
  background: linear-gradient(180deg, #f8fafc 0%, #eefdf7 100%);
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}
.metric-card {
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid rgba(15, 118, 110, 0.18);
  border-radius: 8px;
  padding: 13px 15px;
  box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
}
.metric-title {
  color: #0f766e;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}
.metric-value {
  color: #111827;
  font-size: 27px;
  font-weight: 800;
  line-height: 1.1;
  margin-top: 7px;
}
.metric-note {
  color: #475569;
  font-size: 12px;
  margin-top: 7px;
}
"""


APP_THEME = gr.themes.Soft(primary_hue='emerald', secondary_hue='blue')


with gr.Blocks(title='ShelfSense MVP') as demo:
    gr.Markdown(
        """
        # ShelfSense MVP
        Event-aware grocery demand forecasting and inventory planning for store managers.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            store = gr.Dropdown(choices=stores, value='S002', label='Store')
            family = gr.Dropdown(choices=families, value='beverages', label='Product family')
            horizon = gr.Radio(choices=[7, 14], value=7, label='Planning horizon')
            current_stock = gr.Number(value=320, label='Current stock on hand')
            service_level = gr.Radio(choices=list(SERVICE_LEVELS), value='Balanced', label='Service target')
            case_pack = gr.Number(value=CASE_PACKS['beverages'], label='Case pack size')
        with gr.Column(scale=2):
            brief = gr.Markdown()
            kpis = gr.HTML()

    with gr.Row():
        sales_plot = gr.Plot(label='Demand forecast')
        traffic_plot = gr.Plot(label='Traffic forecast')

    with gr.Row():
        order_table = gr.Dataframe(label='Daily inventory plan', interactive=False)

    with gr.Row():
        forecast_table = gr.Dataframe(label='Forecast detail', interactive=False)

    export_file = gr.File(label='Download selected order plan')

    outputs = [brief, kpis, sales_plot, traffic_plot, order_table, forecast_table, export_file]
    inputs = [store, family, horizon, current_stock, service_level, case_pack]

    for control in [store, horizon, current_stock, service_level, case_pack]:
        control.change(load_view, inputs=inputs, outputs=outputs, queue=False)
    family.change(default_case_pack, inputs=family, outputs=case_pack, queue=False).then(
        load_view, inputs=inputs, outputs=outputs, queue=False
    )
    demo.load(load_view, inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    demo.launch(theme=APP_THEME, css=CSS)
