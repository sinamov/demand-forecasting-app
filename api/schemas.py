# /api/schemas.py

from pydantic import BaseModel, Field
from datetime import date

class ForecastWeek(BaseModel):
    """
    Defines the data structure and validation rules for a single row 
    in the user-uploaded forecast request CSV.
    """
    week: date
    sku_id: int
    store_id: int
    base_price: float
    total_price: float
    is_featured_sku: int = Field(..., ge=0, le=1) # Ensures value is 0 or 1
    is_display_sku: int = Field(..., ge=0, le=1)  # Ensures value is 0 or 1
    
    
    class Config:
        # Allows Pydantic to work seamlessly with other object types
        from_attributes = True