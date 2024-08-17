import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from sklearn.metrics import mean_squared_error, r2_score

from adultcensus_model.config.core import config
from adultcensus_model.pipeline import adultcensus_pipe
from adultcensus_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    print("Training the model -->")
    adultcensus_pipe.fit(X_train,y_train)
    print("Testing the model -->")
    y_pred = adultcensus_pipe.predict(X_test)

    # persist trained model
    save_pipeline(pipeline_to_persist=adultcensus_pipe)
    # calculate and printing the score
    print("Accuracy =", accuracy_score(y_test, y_pred))
    print("Precision =", precision_score(y_test, y_pred))
    print("Recall =", recall_score(y_test, y_pred))
    print("F1 (weighted) =", f1_score(y_test, y_pred, average="weighted"))
    print("F1 (default - binary) =", f1_score(y_test, y_pred))
    print("F1 (macro) =", f1_score(y_test, y_pred, average="macro"))
    print("ROC AUC =", roc_auc_score(y_test, y_pred))    

if __name__ == "__main__":
    run_training()