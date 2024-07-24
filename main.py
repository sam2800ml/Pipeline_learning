from typing import Annotated
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

from zenml import step, pipeline

import logging

logging.basicConfig(level=logging.INFO)

@step
def training_data_loader() -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    logging.info("Loading breast cancer..")
    brest_cancer = load_breast_cancer(as_frame=True)
    logging.info("Splitting train and test..")
    X_train, X_test, y_train, y_test = train_test_split(brest_cancer.data, brest_cancer.target, test_size=0.2, shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test

@step
def svc_trainer(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    gamma: float = 0.001,
) -> Tuple[
    Annotated[ClassifierMixin, "Trained_model"],
    Annotated[float, "Training_acc"],
]:
    model = SVC(gamma=gamma)
    model.fit(X_train.to_numpy(),y_train.to_numpy())
    train_acc = model.score(X_train.to_numpy(), y_train.to_numpy())
    print(f"Train accuracy: {train_acc}")

    return model, train_acc
@step
def svc_prediction(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
) -> Annotated[pd.Series, "Predictions"]:
    predictions = model.predict(X_test.to_numpy())
    return pd.Series(predictions)

@pipeline
def training_pipeline(gamma: float=0.002):
    X_train, X_test, y_train, y_test = training_data_loader()
    model, train_acc = svc_trainer(gamma=gamma, X_train=X_train, y_train=y_train)
    predictions = svc_prediction(model=model, X_test=X_test)
    logging.info(f"Predictions: {predictions}")

if __name__ == "__main__":
    training_pipeline()