# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
from datetime import datetime

# We will continue to use our robust, decoupled prediction and feature engineering functions
from api.prediction import generate_forecast
from api.schemas import ForecastWeek
from pydantic import ValidationError

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Demand Forecasting App",
    page_icon="📈",
    layout="wide"
)

# --- MODEL AND DATA LOADING ---
@st.cache_resource
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
            
    # Get lists of all SKUs and Stores available in the historical data
    available_skus = sorted(list(artifacts.keys()))
    available_stores = sorted(history_df['store_id'].unique().tolist()) if history_df is not None else []
            
    return artifacts, history_df, available_skus, available_stores

# Load the assets. The decorator will cache this so it only runs once.
ARTIFACTS, HISTORY_DF, AVAILABLE_SKUS, AVAILABLE_STORES = load_assets()

# --- USER INTERFACE ---
st.title("📈 Retail Demand Forecasting")
st.write(
    "This app forecasts future product demand based on your inputs. "
    "Use the table below to enter the known data for the weeks you want to forecast."
)

# --- 1. EDITABLE TABLE FOR USER INPUT (Replaces File Upload) ---
st.subheader("Future Week Data Input")

# Create a sample DataFrame to guide the user
initial_data = {
    "week": ["16/07/13", "23/07/13"],
    "sku_id": [AVAILABLE_SKUS[0], AVAILABLE_SKUS[0]],
    "store_id": [AVAILABLE_STORES[0], AVAILABLE_STORES[1]],
    "base_price": [150.0, 150.0],
    "total_price": [145.0, 140.0],
    "is_featured_sku": [0, 1],
    "is_display_sku": [1, 1],
}
input_df = pd.DataFrame(initial_data)

# Use st.data_editor to create an interactive, editable table
edited_df = st.data_editor(
    input_df,
    num_rows="dynamic",
    column_config={
        "week": st.column_config.DateColumn(
            "Week (Start Date)",
            format="DD/MM/YY",
            step=timedelta(days=7),
            required=True,
        ),
        "sku_id": st.column_config.SelectboxColumn(
            "SKU ID",
            options=AVAILABLE_SKUS,
            required=True
        ),
        "store_id": st.column_config.SelectboxColumn(
            "Store ID",
            options=AVAILABLE_STORES,
            required=True
        ),
        "is_featured_sku": st.column_config.CheckboxColumn("Featured?", default=False),
        "is_display_sku": st.column_config.CheckboxColumn("On Display?", default=False),
    },
    key="forecast_input_editor"
)

# --- SIDEBAR FOR FILTERING AND SUBMITTING ---
with st.sidebar:
    st.header("Forecast Parameters")
    
    # Dropdown to select a SKU from the user's input table
    sku_options = sorted(edited_df['sku_id'].unique().tolist())
    selected_sku = st.selectbox(
        "Select a SKU ID to Forecast:",
        options=sku_options
    )
    
    # --- 2. STORE ID DROPDOWN WITH "ALL STORES" ---
    store_options_in_table = sorted(edited_df[edited_df['sku_id'] == selected_sku]['store_id'].unique().tolist())
    store_options = ["All Stores"] + store_options_in_table
    selected_store = st.selectbox(
        "Select a Store ID to Display:",
        options=store_options
    )
    
    # The main action button
    submit_button = st.button("Generate Forecast", type="primary")

# --- MAIN PANEL FOR LOGIC AND OUTPUT ---
if submit_button:
    if edited_df.empty:
        st.warning("Please enter data into the input table before generating a forecast.")
    else:
        try:
            # Prepare the dataframe for prediction
            future_df = edited_df.copy()
            future_df['week'] = pd.to_datetime(future_df['week'], format="DD/MM/YY")
            
            # Pydantic validation (now with a clean error message)
            data_dicts = future_df.to_dict(orient='records')
            [ForecastWeek(**row) for row in data_dicts]
            
            # Filter for the selected SKU before forecasting
            future_df_sku = future_df[future_df['sku_id'] == selected_sku]

            if future_df_sku.empty:
                st.error(f"The input table contains no data for the selected SKU ID: {selected_sku}.")
            else:
                with st.spinner("Generating forecast... This may take a moment."):
                    predictions_df = generate_forecast(ARTIFACTS, HISTORY_DF, future_df_sku)

                st.success("Forecast generated successfully!")
                
                # --- LOGIC FOR "ALL STORES" vs. SINGLE STORE ---
                if selected_store == "All Stores":
                    display_df = predictions_df.groupby('week', as_index=False)['units_sold'].sum()
                    plot_title = f"Total Aggregated Demand for SKU {selected_sku}"
                else:
                    display_df = predictions_df[predictions_df['store_id'] == selected_store].copy()
                    plot_title = f"Demand Forecast for SKU {selected_sku} at Store {selected_store}"

                if display_df.empty:
                    st.warning(f"No forecast data available for the selected store(s).")
                else:
                    # --- DISPLAY THE INTERACTIVE PLOT ---
                    st.subheader(plot_title)
                    display_df['week'] = pd.to_datetime(display_df['week'])
                    fig = px.bar(
                        display_df, x='week', y='units_sold', template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- 3. DOWNLOAD BUTTON (Replaces unwanted text) ---
                    st.subheader("Download Forecast Data")
                    
                    # Convert dataframe to CSV string for the download button
                    csv_data = display_df.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                       label="Download data as CSV",
                       data=csv_data,
                       file_name=f'forecast_sku_{selected_sku}.csv',
                       mime='text/csv',
                    )
                    
                    st.dataframe(display_df)

        except ValidationError:
            st.error("Input data is invalid. Please ensure all required fields in the table are filled correctly.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Fill out the input table and click 'Generate Forecast' in the sidebar.")