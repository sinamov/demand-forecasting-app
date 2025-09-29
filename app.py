# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta, date

from api.prediction import generate_forecast
from api.schemas import ForecastWeek
from pydantic import ValidationError

st.set_page_config(page_title="Demand Forecasting App", page_icon="📈", layout="wide")

@st.cache_resource(show_spinner="Loading models and historical data...")
def load_assets():
    """Loads all artifacts and historical data from the disk."""
    artifacts = {}
    artifacts_dir = 'artifacts'
    history_path = os.path.join(artifacts_dir, 'train_history.csv')
    
    history_df = None
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path, parse_dates=['week'])
    
    for filename in os.listdir(artifacts_dir):
        if filename.startswith('model_bundle_sku_') and filename.endswith('.joblib'):
            sku_id = int(filename.split('_')[3].split('.')[0])
            artifacts[sku_id] = joblib.load(os.path.join(artifacts_dir, filename))
            
    available_skus = sorted(list(artifacts.keys()))
    available_stores = sorted(history_df['store_id'].unique().tolist()) if history_df is not None else []
            
    return artifacts, history_df, available_skus, available_stores

ARTIFACTS, HISTORY_DF, AVAILABLE_SKUS, AVAILABLE_STORES = load_assets()

st.title("📈 Retail Demand Forecasting")
st.write("This app forecasts future product demand. Use the tabs below to either enter data manually or upload a CSV file.")

input_df = None
tab1, tab2 = st.tabs(["Enter Data Manually", "Upload CSV File"])

with tab1:
    st.subheader("Manually Enter Future Week Data")
    st.markdown("_Note: Please ensure all dates selected in the 'Week' column are Mondays._")

    # --- FIX 2: Add a third example row ---
    initial_data = {
        "week": [date(2013, 7, 15), date(2013, 7, 22), date(2013, 7, 29), date(2013, 7, 29)],
        "sku_id": [AVAILABLE_SKUS[0], AVAILABLE_SKUS[0], AVAILABLE_SKUS[1], AVAILABLE_SKUS[0]],
        "store_id": [AVAILABLE_STORES[0], AVAILABLE_STORES[1], AVAILABLE_STORES[0], AVAILABLE_STORES[1]],
        "base_price": [150.0, 150.0, 200.0, 160.0],
        "total_price": [145.0, 140.0, 195.0, 150.0],
        "is_featured_sku": [False, True, False, True],
        "is_display_sku": [True, True, True, False],
    }
    input_df_manual = pd.DataFrame(initial_data)

    min_date = date(2013, 7, 15)
    max_date = date(2013, 9, 30)

    edited_df = st.data_editor(
        input_df_manual, num_rows="dynamic",
        column_config={
            "week": st.column_config.DateColumn("Week (Mondays Only)", help="Only Mondays can be selected.", min_value=min_date, max_value=max_date, format="DD/MM/YYYY", required=True),
            "sku_id": st.column_config.SelectboxColumn("SKU ID", options=AVAILABLE_SKUS, required=True),
            "store_id": st.column_config.SelectboxColumn("Store ID", options=AVAILABLE_STORES, required=True),
            "is_featured_sku": st.column_config.CheckboxColumn("Featured?"),
            "is_display_sku": st.column_config.CheckboxColumn("On Display?"),
        },
        key="forecast_input_editor"
    )

with tab2:
    st.subheader("Upload Future Week Data")
    st.markdown("_Note: Please ensure all dates in your CSV's 'week' column are Mondays._")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file, parse_dates=['week'], date_format='%d/%m/%y')

if input_df is None:
    input_df = edited_df

with st.sidebar:
    st.header("Forecast Parameters")
    
    # --- FIX 1: Convert IDs to integers to remove the '.0' ---
    if not input_df.empty and 'sku_id' in input_df.columns:
        sku_options = sorted(pd.to_numeric(input_df['sku_id'], errors='coerce').dropna().astype(int).unique().tolist())
    else:
        sku_options = []

    if not sku_options:
        selected_sku = st.selectbox("Select a SKU ID:", options=["Enter data first"])
    else:
        selected_sku = st.selectbox("Select a SKU ID:", options=sku_options)
    
    if not input_df.empty and selected_sku and selected_sku != "Enter data first":
        store_options_in_table = sorted(pd.to_numeric(input_df[input_df['sku_id'] == selected_sku]['store_id'], errors='coerce').dropna().astype(int).unique().tolist())
    else:
        store_options_in_table = []
        
    store_options = ["All Stores"] + store_options_in_table
    selected_store = st.selectbox("Select a Store to Display:", options=store_options)
    
    submit_button = st.button("Generate Forecast", type="primary")

if submit_button:
    if input_df.empty:
        st.warning("Please enter or upload data before generating a forecast.")
    else:
        try:
            future_df = input_df.copy()
            future_df['week'] = pd.to_datetime(future_df['week'])

            non_mondays = future_df[future_df['week'].dt.weekday != 0]
            if not non_mondays.empty:
                st.error("Input Error: Please ensure all dates are Mondays.")
                st.stop()

            _ = [ForecastWeek(**row) for row in future_df.to_dict(orient='records')]
            
            future_df_sku = future_df[future_df['sku_id'] == selected_sku]

            if future_df_sku.empty:
                st.error(f"The input data has no rows for the selected SKU ID: {selected_sku}.")
            else:
                with st.spinner("Generating forecast..."):
                    predictions_df = generate_forecast(ARTIFACTS, HISTORY_DF, future_df_sku)

                st.success("Forecast generated successfully!")
                
                if selected_store == "All Stores":
                    display_df = predictions_df.groupby('week', as_index=False)['units_sold'].sum()
                    plot_title = f"Total Aggregated Demand for SKU {selected_sku}"
                else:
                    display_df = predictions_df[predictions_df['store_id'] == selected_store].copy()
                    plot_title = f"Demand Forecast for SKU {selected_sku} at Store {selected_store}"

                if display_df.empty:
                    st.warning(f"No forecast data available for the selected store(s).")
                else:
                    display_df = display_df.rename(columns={'units_sold': 'demand (#)'})
                    st.subheader(plot_title)
                    display_df['week'] = pd.to_datetime(display_df['week'])
                    
                    display_df['bar_center'] = display_df['week'] + pd.to_timedelta(3, unit='D')
                    
                    min_plot_date = display_df['week'].min()
                    max_plot_date = display_df['week'].max()
                    start_range = min_plot_date - pd.to_timedelta(1, unit='D')
                    end_range = max_plot_date + pd.to_timedelta(7, unit='D')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=display_df['bar_center'], y=display_df['demand (#)'],
                        width=6 * 24 * 60 * 60 * 1000, name='Forecast',
                        hovertemplate='Week of %{customdata|%b %d, %Y}<br>Demand: %{y}<extra></extra>',
                        customdata=display_df['week']
                    ))

                    fig.update_layout(
                        title_text=plot_title, xaxis_title="Week", yaxis_title="Number of Items",
                        template='plotly_white', bargap=0.1, xaxis_range=[start_range, end_range]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Download Forecast Data")
                    output_columns = ['week', 'sku_id', 'store_id', 'demand (#)']
                    final_output_df = display_df[[col for col in output_columns if col in display_df.columns]]
                    csv_data = final_output_df.to_csv(index=False, date_format='%Y-%m-%d').encode('utf-8')
                    st.download_button(
                       label="Download data as CSV", data=csv_data,
                       file_name=f'forecast_{selected_sku}_{selected_store}.csv', mime='text/csv'
                    )
                    
                    df_for_display = final_output_df.copy()
                    df_for_display['week'] = df_for_display['week'].dt.strftime('%Y-%m-%d')
                    
                    st.dataframe(df_for_display)

        except ValidationError as e:
            st.error(f"Input data is invalid: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e)