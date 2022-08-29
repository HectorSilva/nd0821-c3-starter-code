"""Train model code
"""
# Script to train machine learning model.
import os
import pickle
from typing import Any

import joblib
# Add the necessary imports for the starter code.
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from .constants import CAT_FEATURES
from .ml.data import process_data
from .ml.model import train_model

cat_features = CAT_FEATURES
abs_path = os.path.abspath(os.path.dirname(__file__))


def get_data():
    """
    The get_data function loads in the data and returns it as a pandas DataFrame.

    :returns: A pandas DataFrame containing the census data.
    """
    print(f'Current directory {os.getcwd()}')
    # Add code to load in the data.
    data = pd.read_csv(os.path.join(abs_path, "../data/census_updated.csv"))
    return data


def train_split(data: DataFrame):
    """
    The train_split function splits the data into a training and testing set.
    The train_split function takes in a DataFrame, splits it into X_train, y_train, encoder and label binarizer.


    :param data:DataFrame: Specify the data to be used for training
    :return: 4 values:
    """
    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train, encoder, lb, test


def process_test_data(test, encoder, lb):
    """
    The process_test_data function takes in a dataframe and returns the processed features, labels, encoder
    and label binarizer for the test set. The function also takes in categorical_features which is a list of
    column names that are categorical variables. The function also takes in label which is the name of the target
    variable and encoder which is an object from sklearn's LabelEncoder class. Finally, it also takes in lb which
    is an object from sklearn's LabelBinarizer class.

    :param test: Determine whether the data is for training or testing
    :param encoder: Encode the categorical features
    :param lb: Transform the labels into a format that can be used by the neural network
    :return: The test data with the same processing as the train data
    """
    # Proces the test data with the process_data function.
    test_X_train, test_y_train, test_encoder, test_lb = process_data(
        test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False)
    return test_X_train, test_y_train, test_encoder, test_lb


# Train and save a model.
def train_ml_model(X_train, y_train):
    """
    The train_ml_model function trains a machine learning model using the training data.
    It returns the trained model object.

    :param X_train: Pass the training data for the model to train on
    :param y_train: Pass the target variable to the train_model function
    :return: The trained model
    """
    model = train_model(X_train=X_train, y_train=y_train)
    return model


def save_artifact(artifact, model_dir: str, file_name: str):
    """
    The save_artifact function saves the artifact passed to it as a pickle file in the model_dir directory.
    The save_artifact function takes two arguments:
        1) The artifact to be saved (e.g., a trained model, or other data).
        2) The name of the file that will be created and used for saving.

    :param artifact: Save the model, which is a python object
    :param model_dir:str: Specify the folder in which to save the artifact
    :param file_name:str: Specify the name of the artifact file
    :return: A dictionary with the following keys:
    """
    with open(f'{abs_path}/{model_dir}/{file_name}', 'wb') as file:
        pickle.dump(artifact, file)


def get_artifact(dir_name: str, artifact_name: str) -> Any:
    """
    The get_artifact function loads in the artifact from a file.

    :param dir_name:str: Specify the directory where the artifact is located
    :param artifact_name:str: Specify the name of the artifact to be loaded
    :return: A file object
    """
    # Add code to load in the artifact.
    artifact = joblib.load(open(f'{abs_path}/{dir_name}/{artifact_name}', 'r+b'))
    return artifact
