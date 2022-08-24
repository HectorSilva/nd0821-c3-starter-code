# Put the code for your API here.
import os
import pickle

import uvicorn
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.fields import Field
from starter.ml.model import inference
from starter.train_model import model
from starter.train_model import data
import pandas as pd
import numpy as np

app = FastAPI()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# sav_trained_model = '~/Documents/Udacity/Module3/nd0821-c3-starter-code/starter/model/trained_model.sav'
# with open(sav_trained_model, 'rb') as file:
#     sav_model = pickle.load(file)


class TaggedPerson(BaseModel):
    age: str
    workclass: str
    education: str  # Union[str, list]
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True


@app.get('/')
async def say_hello():
    return {"greeting": "Welcome to my MLOps project!"}


@app.post('/ingest/')
async def ingest_person(item: TaggedPerson):
    return item


@app.post('/inference/')
async def get_inference(data: TaggedPerson):
    input_df = pd.DataFrame([data])
    print('Pandas array', data, type(data), [data], 'Data frame', input_df)
    print(f'model type {type(model)}')

    input_data = pd.DataFrame([['24', '39'],
                               ['12', '23'],
                               ['31', '13'],
                               ['22', '42'],
                               ['43', '33'],
                               ['34', '23'],
                               ['12', '43'],
                               ['1', '33'],
                               ['32', '43']], dtype=str)

    pred = inference(model, np.array(input_data))
    return pred


@app.get('/people/{item_id}')
async def get_items(item_id: int, count: int = 1):
    inference(model, data)
    return {"fetch": f"Fetched {count} of {item_id}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
