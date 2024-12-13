import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from zipfile import ZipFile
import os
import numpy as np

def generateFiles(predict_data, clf_pipe):
    """Genera los archivos a subir en CodaLab

    Input
    ---------------
    predict_data: Dataframe con los datos de entrada a predecir
    clf_pipe: pipeline del clf

    Ouput
    ---------------
    archivo de txt
    """
    y_pred_clf = clf_pipe.predict_proba(predict_data)[:, 1]
    with open('./predictions.txt', 'w') as f:
        for item in y_pred_clf:
            f.write("%s\n" % item)
    
    with ZipFile('predictions.zip', 'w') as zipObj:
        zipObj.write('predictions.txt')
    os.remove('predictions.txt')

def process_timestamps(X):
    X = X.copy()
    X["borrow_timestamp"] = pd.to_datetime(X["borrow_timestamp"], unit="s")
    X["first_tx_timestamp"] = pd.to_datetime(X["first_tx_timestamp"], unit="s")
    X["last_tx_timestamp"] = pd.to_datetime(X["last_tx_timestamp"], unit="s")
    X["borrow_year"] = X["borrow_timestamp"].dt.year
    X["borrow_month"] = X["borrow_timestamp"].dt.month
    X["first_tx_year"] = X["first_tx_timestamp"].dt.year
    X["first_tx_month"] = X["first_tx_timestamp"].dt.month
    X["last_tx_year"] = X["last_tx_timestamp"].dt.year
    X["last_tx_month"] = X["last_tx_timestamp"].dt.month
    X = X.drop(["borrow_timestamp", "first_tx_timestamp", "last_tx_timestamp", "risky_first_tx_timestamp", "risky_last_tx_timestamp"], axis=1)
    return X

def define_columns(X):
    categorical_features = ["first_tx_year", "first_tx_month", "last_tx_year", "last_tx_month", "borrow_year", "borrow_month"]
    numeric_features = [col for col in X.columns if col not in categorical_features]
    return numeric_features, categorical_features

def identity_transform(X):
    return X

class DropHighlyCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9, exclude_columns=None):
        """
        Clase para eliminar columnas altamente correlacionadas.

        Parameters:
        - threshold: umbral de correlación para eliminar columnas (por defecto 0.9).
        - exclude_columns: lista de columnas que no deben eliminarse (por defecto None).
        """
        self.threshold = threshold
        self.columns_to_drop = []
        self.exclude_columns = exclude_columns if exclude_columns else []

    def fit(self, X, y=None):
        correlation_matrix = X.corr().abs()
        to_drop = set()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > self.threshold:
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]

                    if col2 not in self.exclude_columns:
                        to_drop.add(col2)

        self.columns_to_drop = list(to_drop)
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")
    
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1, errors='ignore')
    

class ReplaceOutliersWithMedian(BaseEstimator, TransformerMixin):
    def __init__(self, method="IQR", factor=1.5):
        self.method = method
        self.factor = factor
        self.medians = None
        self.lower_bounds = None
        self.upper_bounds = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.medians = X.median()
        if self.method == "IQR":
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds = Q1 - self.factor * IQR
            self.upper_bounds = Q3 + self.factor * IQR
        else:
            raise ValueError("Método no soportado. Usa 'IQR'.")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()
        for col in X.columns:
            if col in self.medians.index:
                X[col] = np.where(
                    (X[col] < self.lower_bounds[col]) | (X[col] > self.upper_bounds[col]),
                    self.medians[col],
                    X[col]
                )
        return X