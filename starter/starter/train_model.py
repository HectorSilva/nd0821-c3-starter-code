"""Train model code
"""
# Script to train machine learning model.
import os
import sys
import pickle
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model

print(f'Current directory { os.getcwd() }')
sys.path.insert(1, '../data/')
# Add code to load in the data.
data = pd.read_csv(
    f'{os.getcwd()}/../data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

test_X_train, test_y_train, test_encoder, test_lb = process_data(
    test, categorical_features=cat_features, label='salary', training=True)

# Train and save a model.

model = train_model(X_train=X_train, y_train=y_train)

filename = 'trained_model.sav'

print(os.getcwd())

with open('../model/' + filename, 'wb') as file:
    pickle.dump(model, file)

print('Finished')
