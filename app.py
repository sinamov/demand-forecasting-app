# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
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
            
    return artifacts, history_df

# Load the assets. The decorator will cache this so it only runs once.
ARTIFACTS, HISTORY_DF = load_assets()
AVAILABLE_SKUS = sorted(list(ARTIFACTS.keys()))

# --- USER INTERFACE ---
st.title("📈 Retail Demand Forecasting")
st.write("Upload a CSV file with future week data to generate a forecast.")

# --- SIDEBAR FOR USER INPUTS ---
with st.sidebar:
    st.header("Forecast Parameters")
    
    # Dropdown to select a valid SKU
    selected_sku = st.selectbox(
        "Select a SKU ID:",
        options=AVAILABLE_SKUS
    )
    
    # Number input for the Store ID
    selected_store = st.number_input(
        "Enter a Store ID:",
        min_value=0,
        step=1,
        value=8091 # A default value
    )
    
    # File uploader for the future data
    uploaded_file = st.file_uploader(
        "Upload your future weeks data (CSV)",
        type="csv"
    )

# --- MAIN PANEL FOR LOGIC AND OUTPUT ---
if uploaded_file is not None:
    try:
        # Load and validate the uploaded data
        future_df = pd.read_csv(uploaded_file, parse_dates=['week'], date_format='%d/%m/%y')
        
        # Validation Block
        data_dicts = future_df.to_dict(orient='records')
        [ForecastWeek(**row) for row in data_dicts]

        # Filter for the selected SKU
        future_df_sku = future_df[future_df['sku_id'] == selected_sku]

        if future_df_sku.empty:
            st.error(f"The uploaded file contains no data for the selected SKU ID: {selected_sku}. Please check your file.")
        else:
            with st.spinner("Generating forecast... This may take a moment."):
                # Generate the forecast
                predictions_df = generate_forecast(ARTIFACTS, HISTORY_DF, future_df_sku)
                
                # Filter for the selected store to plot
                plot_df = predictions_df[predictions_df['store_id'] == selected_store].copy()

            st.success("Forecast generated successfully!")

            if plot_df.empty:
                st.warning(f"A forecast was generated for SKU {selected_sku}, but no predictions were made for the selected Store ID: {selected_store}.")
            else:
                # --- DISPLAY THE INTERACTIVE PLOT ---
                st.subheader(f"Forecast for SKU {selected_sku} at Store {selected_store}")
                
                plot_df['week'] = pd.to_datetime(plot_df['week'])
                
                fig = px.bar(
                    plot_df, 
                    x='week', 
                    y='units_sold', 
                    title=f'Weekly Demand Forecast', 
                    template='plotly_white'
                )
                
                # Use Streamlit's native Plotly chart element
                st.plotly_chart(fig, use_container_width=True)
                
                # --- DISPLAY THE DATA TABLE ---
                st.subheader("Forecast Data")
                st.dataframe(plot_df)

    except ValidationError as e:
        st.error(f"Invalid CSV format: {e.errors()}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")