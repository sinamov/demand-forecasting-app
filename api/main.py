# /api/main.py

import pandas as pd
import joblib
import plotly.express as px
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import ValidationError
import io
import os

from .prediction import generate_forecast
from .schemas import ForecastWeek

app = FastAPI(title="Demand Forecasting API")

ARTIFACTS = {}
HISTORY_DF = None

@app.on_event("startup")
def load_artifacts():
    global HISTORY_DF, ARTIFACTS
    artifacts_dir = 'artifacts'
    history_path = os.path.join(artifacts_dir, 'train_history.csv')
    
    if os.path.exists(history_path):
        HISTORY_DF = pd.read_csv(history_path, parse_dates=['week'])
    
    for filename in os.listdir(artifacts_dir):
        if filename.startswith('model_bundle_sku_') and filename.endswith('.joblib'):
            sku_id = int(filename.split('_')[3].split('.')[0])
            ARTIFACTS[sku_id] = joblib.load(os.path.join(artifacts_dir, filename))
    print(f"--- Artifact loading complete. Found {len(ARTIFACTS)} bundles. ---")

@app.post("/forecast/file", tags=["Forecasting"])
async def get_forecast_csv(sku_id: int, file: UploadFile = File(...)):
    if sku_id not in ARTIFACTS:
        raise HTTPException(status_code=404, detail=f"No model found for SKU {sku_id}")
    
    try:
        future_df = pd.read_csv(file.file, parse_dates=['week'], date_format='%d/%m/%y')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV. Ensure dates are in DD/MM/YY format. Error: {e}")

    try:
        data_dicts = future_df.to_dict(orient='records')
        [ForecastWeek(**row) for row in data_dicts]
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid data format in CSV file: {e.errors()}")

    # --- FIX: Filter the uploaded data to match the requested SKU ID ---
    original_row_count = len(future_df)
    future_df = future_df[future_df['sku_id'] == sku_id]

    if future_df.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"The uploaded CSV file has {original_row_count} rows, but none match the requested sku_id: {sku_id}."
        )

    predictions_df = generate_forecast(ARTIFACTS, HISTORY_DF, future_df)
    
    output_buffer = io.StringIO()
    predictions_df.to_csv(output_buffer, index=False)
    output_buffer.seek(0)
    
    return StreamingResponse(
        output_buffer,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=forecast_sku_{sku_id}.csv"}
    )

@app.post("/forecast/plot", tags=["Forecasting"], response_class=HTMLResponse)
async def get_forecast_plot(sku_id: int, store_id: int, file: UploadFile = File(...)):
    # (The same fix is applied here)
    if sku_id not in ARTIFACTS:
        raise HTTPException(status_code=404, detail=f"No model found for SKU {sku_id}")
        
    try:
        future_df = pd.read_csv(file.file, parse_dates=['week'], date_format='%d/%m/%y')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV. Ensure dates are in DD/MM/YY format. Error: {e}")

    try:
        data_dicts = future_df.to_dict(orient='records')
        [ForecastWeek(**row) for row in data_dicts]
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid data format in CSV file: {e.errors()}")
        
    # --- FIX: Filter the uploaded data to match the requested SKU ID ---
    original_row_count = len(future_df)
    future_df = future_df[future_df['sku_id'] == sku_id]
    
    if future_df.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"The uploaded CSV file has {original_row_count} rows, but none match the requested sku_id: {sku_id}."
        )

    predictions_df = generate_forecast(ARTIFACTS, HISTORY_DF, future_df)
    plot_df = predictions_df[predictions_df['store_id'] == store_id].copy()
    
    
    if plot_df.empty:
        raise HTTPException(status_code=404, detail=f"Forecast generated, but no data found for the specific store_id: {store_id}.")
    
    plot_df['week'] = pd.to_datetime(plot_df['week'])
    
    # 2. Create a NEW column containing the dates formatted as strings.
    #    '%b %d, %Y' will format the date like "Sep 27, 2025".
    plot_df['week_formatted'] = plot_df['week'].dt.strftime('%b %d, %Y')
    
    # 3. Tell Plotly to use the new STRING column for the x-axis.
    fig = px.bar(
        plot_df,
        x='week_formatted', # Use the new formatted string column here
        y='units_sold',
        title=f'Weekly Demand Forecast for SKU {sku_id} at Store {store_id}',
        template='plotly_white'
    )
    
    # The tickformat command is no longer needed as we are providing strings.
    # We can use update_layout for better sorting if needed.
    fig.update_layout(xaxis={'categoryorder':'total descending'})

    html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # Return the HTML content using FastAPI's HTMLResponse.
    return HTMLResponse(content=html_content)