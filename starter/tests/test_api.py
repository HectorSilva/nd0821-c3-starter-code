import os
import sys

from fastapi.testclient import TestClient

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURR_DIR + '/../../')

from starter.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Welcome to my MLOps project!"}


def test_correct_payload():
    payload = {
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
    response = client.post("/predict/", headers={'Content-Type': 'application/json'}, json=payload)
    assert response.status_code == 200
    assert {"Salary": ">50k"} == response.json()


def test_incorrect_payload():
    payload = {
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
        "native-country43": "United-States"
    }
    expected_res = {'detail': [{'loc': ['body', 'native-country'],
                                'msg': 'field required',
                                'type': 'value_error.missing'}]}
    response = client.post("/predict/", headers={'Content-Type': 'application/json'}, json=payload)
    assert response.status_code == 422
    assert expected_res == response.json()


def test_empty_payload():
    payload = {}
    response = client.post("/predict/", headers={'Content-Type': 'application/json'}, json=payload)
    assert response.status_code == 422
