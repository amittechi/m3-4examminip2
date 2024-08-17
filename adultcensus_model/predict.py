import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from adultcensus_model import __version__ as _version
from adultcensus_model.config.core import config
from adultcensus_model.pipeline import adultcensus_pipe
from adultcensus_model.processing.data_manager import load_pipeline
from adultcensus_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
adultcensus_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    validated_data=validated_data.reindex(columns=config.model_config.features)
    
    print("validated_data\n", validated_data)
    results = {"validated_data before predict": None, "version": _version, "errors": errors}
    
    print(results)
    if not errors:

        predictions = adultcensus_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        print(results)

    return results

if __name__ == "__main__":


    data_in = {
        'age': [37],
        'workclass': ['Private'],
        'fnlwgt': [284582],
        'education': ['Bachelors'],
        'education-num': [13],
        'marital-status': ['Married-civ-spouse'],
        'occupation': ['Exec-managerial'],
        'relationship': ['Husband'],
        'race': ['White'],
        'sex': ['Male'],
        'capital-gain': [0],
        'capital-loss': [0],
        'hours-per-week': [50],
        'native-country': ['United-States'],
        'class': ['>50K']
    }
    
    # data_in = {
    #     'age': [45],
    #     'workclass': ['Self-emp-not-inc'],
    #     'fnlwgt': [209642],
    #     'education': ['HS-grad'],
    #     'education-num': [9],
    #     'marital-status': ['Divorced'],
    #     'occupation': ['Sales'],
    #     'relationship': ['Not-in-family'],
    #     'race': ['Black'],
    #     'sex': ['Female'],
    #     'capital-gain': [0],
    #     'capital-loss': [0],
    #     'hours-per-week': [40],
    #     'native-country': ['United-States'],
    #     'class': ['<=50K']
    # },
    # {
    #     'age': 29,
    #     'workclass': 'Private',
    #     'fnlwgt': 150039,
    #     'education': 'Masters',
    #     'education-num': 14,
    #     'marital-status': 'Never-married',
    #     'occupation': 'Prof-specialty',
    #     'relationship': 'Not-in-family',
    #     'race': 'Asian-Pac-Islander',
    #     'sex': 'Female',
    #     'capital-gain': 0,
    #     'capital-loss': 0,
    #     'hours-per-week': 45,
    #     'native-country': 'India',
    #     'class': '>50K'
    # },
    # {
    #     'age': 53,
    #     'workclass': 'Federal-gov',
    #     'fnlwgt': 182148,
    #     'education': 'Doctorate',
    #     'education-num': 16,
    #     'marital-status': 'Married-civ-spouse',
    #     'occupation': 'Prof-specialty',
    #     'relationship': 'Husband',
    #     'race': 'White',
    #     'sex': 'Male',
    #     'capital-gain': 14084,
    #     'capital-loss': 0,
    #     'hours-per-week': 60,
    #     'native-country': 'United-States',
    #     'class': '>50K'
    # },
    # {
    #     'age': 31,
    #     'workclass': 'Private',
    #     'fnlwgt': 344300,
    #     'education': 'Assoc-acdm',
    #     'education-num': 12,
    #     'marital-status': 'Married-civ-spouse',
    #     'occupation': 'Craft-repair',
    #     'relationship': 'Husband',
    #     'race': 'White',
    #     'sex': 'Male',
    #     'capital-gain': 0,
    #     'capital-loss': 0,
    #     'hours-per-week': 40,
    #     'native-country': 'United-States',
    #     'class': '<=50K'
    # },
    # {
    #     'age': 22,
    #     'workclass': 'Private',
    #     'fnlwgt': 238271,
    #     'education': 'Some-college',
    #     'education-num': 10,
    #     'marital-status': 'Never-married',
    #     'occupation': 'Adm-clerical',
    #     'relationship': 'Own-child',
    #     'race': 'White',
    #     'sex': 'Female',
    #     'capital-gain': 0,
    #     'capital-loss': 0,
    #     'hours-per-week': 30,
    #     'native-country': 'United-States',
    #     'class': '<=50K'
    # }   
    make_prediction(input_data=data_in)
