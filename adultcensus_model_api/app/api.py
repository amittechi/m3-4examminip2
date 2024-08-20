import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
import io
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
from fastapi.encoders import jsonable_encoder
from adultcensus_model import __version__ as model_version
from adultcensus_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Adult Census Class predictions with the adultcensus_model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results

# New /upload endpoint
@api_router.post("/upload", response_class=PlainTextResponse)
async def upload(file: UploadFile = File(...)) -> str:
    """
    Upload CSV, predict line by line, and return the CSV with predictions.
    """
    try:
        # Read the uploaded CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV file: {e}")

    # Prepare a list to collect rows with predictions
    predicted_rows = []

    for index, row in df.iterrows():
        try:
             # Convert row to the appropriate schema, ensuring empty strings for NaNs
            row_dict = {k: (v if pd.notna(v) else "") for k, v in row.to_dict().items()}
            input_data = schemas.MultipleDataInputs(inputs=[row_dict])

            # Call the predict function
            input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
            results = make_prediction(input_data=input_df.replace({np.nan: None}))

            if results["errors"] is not None:
                raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

            # Extract the prediction value from the array
            prediction = results["predictions"]
            if isinstance(prediction, np.ndarray) and prediction.size == 1:
                prediction = prediction.item()
            row_dict["prediction"] = prediction
            predicted_rows.append(row_dict)

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction failed for row {index}: {e}")

    # Convert the list of predicted rows back to a DataFrame
    predicted_df = pd.DataFrame(predicted_rows)

    # Convert the DataFrame to CSV
    output = io.StringIO()
    predicted_df.to_csv(output, index=False)
    output.seek(0)
    return output.read()