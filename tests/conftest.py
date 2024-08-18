import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from sklearn.model_selection import train_test_split

from adultcensus_model.config.core import config
from adultcensus_model.processing.data_manager import load_dataset


@pytest.fixture
def sample_input_data():
    # we will run tests on unseen data of 16.5K records
    #data = load_dataset(file_name=config.app_config.training_data_file)
    data = load_dataset(file_name=config.app_config.test_data_file)

    X = data.drop(config.model_config.target, axis=1)       # predictors
    y = data[config.model_config.target]                    # target

    # we will not divide this dataset into train and test and let X_test be X and y_test be y
    # divide train and test
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,  # predictors
    #     y,  # target
    #     test_size=config.model_config.test_size,
    #     # we are setting the random seed here
    #     # for reproducibility
    #     random_state=config.model_config.random_state,
    # )
    
    X_test = X
    y_test = y

    return X_test, y_test