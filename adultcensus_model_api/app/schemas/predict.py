from typing import Any, List, Optional

from pydantic import BaseModel
from adultcensus_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        'age': 31,
                        'workclass': 'Private',
                        'fnlwgt': 344300,
                        'education': 'Assoc-acdm',
                        'education-num': 12,
                        'marital-status': 'Married-civ-spouse',
                        'occupation': 'Craft-repair',
                        'relationship': 'Husband',
                        'race': 'White',
                        'sex': 'Male',
                        'capital-gain': 0,
                        'capital-loss': 0,
                        'hours-per-week': 40,
                        'native-country': 'United-States',
                        'class': '<=50K'
                    }
                ]
            }
        }
