import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datetime import date
from pydantic import BaseModel, ValidationError, Field

from adultcensus_model.config.core import config
from adultcensus_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

 
class DataInputSchema(BaseModel):
    age: Optional[float]
    workclass: Optional[str]
    fnlwgt: Optional[float]
    education: Optional[str]
    education_num: Optional[float] = Field(None, alias="education-num")
    marital_status: Optional[str] = Field(None, alias="marital-status")
    occupation: Optional[str]
    relationship: Optional[str]
    race: Optional[str]
    sex: Optional[str]
    capital_gain: Optional[float] = Field(None, alias="capital-gain")
    capital_loss: Optional[float] = Field(None, alias="capital-loss")
    hours_per_week: Optional[float] = Field(None, alias="hours-per-week")
    native_country: Optional[str] = Field(None, alias="native-country")
    income_class: Optional[str] = Field(None, alias="class")

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]