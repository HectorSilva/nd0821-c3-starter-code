import os
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from starter.constants import CAT_FEATURES
from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import get_artifact, get_data, train_split


class ModelTest(TestCase):

    def _setup(self):
        abs_path = os.path.abspath(os.path.dirname(__file__))
        model_dir = os.path.join(abs_path, '../model')
        model = get_artifact(model_dir, 'trained_model.sav')
        encoder = get_artifact(model_dir, 'onehot_encoder.sav')
        lb = get_artifact(model_dir, 'lb.sav')
        return model, encoder, lb

    def predict(self, payload) -> int:
        model, encoder, lb = self._setup()
        df = pd.DataFrame(payload, index=[0])
        X, _, _, _ = process_data(df, categorical_features=CAT_FEATURES, lb=lb,
                                  encoder=encoder,
                                  training=False)
        pred = inference(model, X)
        return X, pred

    def test_model_lower_salary(self):
        higher_salary_payload = {
            "age": "52",
            "workclass": "Self-emp-not-inc",
            "fnlgt": 209642,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Never-married",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 14084,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }
        X, pred = self.predict(higher_salary_payload)
        self.assertEqual(type(X), np.ndarray)
        self.assertEqual(pred, 1)

    def test_model_higher_salary(self):
        lower_salary_payload = {
            "age": "26",
            "workclass": "Self-emp-not-inc",
            "fnlgt": 209642,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Widow",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Female",
            "capital-gain": 184,
            "capital-loss": 300,
            "hours-per-week": 43,
            "native-country": "Cuba"
        }

        X, pred = self.predict(lower_salary_payload)
        self.assertEqual(type(X), np.ndarray)
        self.assertEqual(pred, 0)

    def test_get_data(self):
        data = get_data()
        self.assertEqual(type(data), DataFrame)

    def test_train_split(self):
        data = get_data()
        X_train, y_train, encoder, lb, _ = train_split(data)
        self.assertEqual(type(X_train), np.ndarray)
        self.assertEqual(type(y_train), np.ndarray)
        self.assertEqual(type(encoder), OneHotEncoder)
        self.assertEqual(type(lb), LabelBinarizer)

    def test_get_artifact(self):
        model, encoder, lb = self._setup()
        self.assertEqual(type(model), RandomForestClassifier)
        self.assertEqual(type(encoder), OneHotEncoder)
        self.assertEqual(type(lb), LabelBinarizer)
