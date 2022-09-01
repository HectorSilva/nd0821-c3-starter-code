import requests

payload = {
    "age": 52,
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
result = requests.post('https://udacity-project3-mlapi.herokuapp.com/predict/',
                       headers={'Content-Type': 'application/json'},
                       json=payload)
print(f'Result in json format: {result.content}, status code: {result.status_code}')
