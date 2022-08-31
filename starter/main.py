# Put the code for your API here.
import os
import subprocess
import time

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.fields import Field

from starter.constants import CAT_FEATURES
from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import get_artifact

app = FastAPI()


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = '-' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = '-' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc config core.hardlink_lock true")

    if 'starter' not in os.getcwd():
        print(f'Working dir {os.getcwd()}', os.listdir('.'))
        os.chdir('starter')
        print(f'Changed to {os.getcwd()} directory')
    else:
        print('Already in starter directory')

    if os.path.exists("../.dvc/tmp/lock") or os.path.exists("../.dvc/tmp/rwlock"):
        lock = os.system('rm ../.dvc/tmp/lock')
        rwlock = os.system('rm ../.dvc/tmp/rwlock.lock')
        print('The lock files have been deleted')
    else:
        print(f'Working dir {os.getcwd()}', os.listdir('.'))
        print('Lock not found')

# print('************** BEGINNING OF LISTING FILES *****************')
# list_files(os.getcwd())
# print('************** END OF LISTING FILES ***********************')

status_code = -1

if 'starter' not in os.getcwd() and os.path.isdir('starter'):
    os.chdir('starter')
    print(f'Changed to {os.getcwd()} directory')
    status_code = subprocess.call(
        ["dvc", "pull", "-r", "s3remote", "-vvv"], timeout=60)
else:
    print(f'No change of directory, running dvc process, curr working dir {os.getcwd()}')
    status_code = subprocess.call(
        ["dvc", "pull", "-r", "s3remote", "-vvv"], timeout=60)
    # print(f'Lock info {lock}, lock type {type(lock)}')

    # status_code = os.system("dvc pull -r s3remote", )

    if status_code != 0:
        print(type(status_code), status_code)
        print(f'Wokring directory: {os.getcwd()}')
        print('Error trying to pull the data, trying again')
        time.sleep(10)
        # print(f'Std errors: {dvc_output.stderr}')
        # print(f'Error code: {dvc_output.returncode}')
        # print(f'ST output: {dvc_output.stdout}')
        # exit("dvc pull failed")

    # os.system("rm -r .dvc ../.apt/usr/lib/dvc")

abs_path = os.path.abspath(os.path.dirname(__file__))
model_dir = os.path.join(abs_path, 'model')
model = get_artifact(model_dir, 'trained_model.sav')
encoder = get_artifact(model_dir, 'onehot_encoder.sav')
lb = get_artifact(model_dir, 'lb.sav')


class Person(BaseModel):
    age: str = Field(example=52)
    workclass: str = Field(example="Self-emp-not-inc")
    fnlgt: int = Field(example=209642)
    education: str = Field(example="Masters")  # Union[str, list]
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str = Field(example='Exec-managerial')
    relationship: str = Field(example='Husband')
    race: str = Field(example='Black')
    sex: str = Field(example='Male')
    capital_gain: int = Field(alias='capital-gain', example=40)
    capital_loss: int = Field(alias='capital-loss', example=0)
    hours_per_week: int = Field(alias='hours-per-week', example=40)
    native_country: str = Field(alias="native-country", example='Mexico')

    class Config:
        allow_population_by_field_name = True


@app.get('/')
async def say_hello():
    """
    The say_hello function returns a greeting.

    :return: A dictionary with a message key and value
    """
    return {"msg": "Welcome to my MLOps project!"}


@app.post('/predict/')
async def get_inference(data: Person):
    """
    The get_inference function takes a Person object and returns the predicted salary.

    :param data:Person: Pass the data to the function
    :return: A dictionary with a single key &quot;salary&quot; and a value &quot;&lt;=50k&quot; or &quot;&gt;50k&quot;
    """
    person_dict = data.dict(by_alias=True)
    person_df = pd.DataFrame(person_dict, index=[0])
    X, _, _, _ = process_data(person_df, categorical_features=CAT_FEATURES, lb=lb,
                              encoder=encoder,
                              training=False)
    pred = inference(model, X)
    return {"Salary": '<=50K' if pred == 0 else '>50k'}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
