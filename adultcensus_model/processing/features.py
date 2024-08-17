import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class ModeImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in category field with mode value of the col """

    def __init__(self, variables: list):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column in self.variables:
            mode_value = X[column].mode()[0]
            X[column].fillna(mode_value, inplace=True)
            print(f"{column} has been imputed!")
        return X
    
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: list):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column in self.variables:
            if not pd.api.types.is_numeric_dtype(X[column]):
                # Handle NaN values before creating the Categorical
                unique_categories = []
                if (X[column].isnull().sum() > 0):
                    X[column] = X[column].fillna(X[column].mode()[0], inplace=True) 
                    unique_categories = X[column].unique().tolist()
                    print(unique_categories)
                X[column] = pd.Categorical(X[column], categories=unique_categories, ordered=True)
                X[column] = X[column].cat.codes
                print(f"{column} has been mapped!")
        return X
    
class CustomMapper(BaseEstimator, TransformerMixin):
    """
    Binary encoding value from mapping since predict class will send only 1 record
    """
    def __init__(self, variables: str, mappings: dict):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        if not isinstance(mappings, dict):
            raise ValueError("mappings should be a dict")

        self.variables = variables
        self.mappings = mappings
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X[self.variables] = X[self.variables].map(self.mappings)
        print(self.variables, 'mapped!')
        # print(X)
        return X

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, variables: list, lower_quantile: float = 0.25, upper_quantile: float = 0.75):
        if not isinstance(variables, list):
            raise ValueError("variable should be list of strings")

        if not all(isinstance(var, str) for var in variables):
            raise ValueError("all variables should be strings")

        if not (0 <= lower_quantile <= 1 and 0 <= upper_quantile <= 1):
            raise ValueError("quantiles should be between 0 and 1")

        self.variables = variables
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.bounds_ = {}
        for var in self.variables:
            lower_bound = X[var].quantile(self.lower_quantile)
            upper_bound = X[var].quantile(self.upper_quantile)
            self.bounds_[var] = (lower_bound, upper_bound)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for var in self.variables:
            lower_bound, upper_bound = self.bounds_[var]
            X[var] = np.where(X[var] > upper_bound, upper_bound, X[var])
            X[var] = np.where(X[var] < lower_bound, lower_bound, X[var])
        print('Outlier removed')
        return X
    
# Define a function to drop specified columns
def drop_columns(X, col_to_drop):
    print("Dropping columns")
    return X.drop(columns=col_to_drop)

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode specified categorical columns """

    def __init__(self, variables: list):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.encoder.fit(X[self.variables])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        one_hot_encoded = self.encoder.transform(X[self.variables])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=self.encoder.get_feature_names_out(self.variables), index=X.index)
        X = X.drop(columns=self.variables)
        X = pd.concat([X, one_hot_encoded_df], axis=1)
        print(f'{self.variables} onehot encoded')
        return X