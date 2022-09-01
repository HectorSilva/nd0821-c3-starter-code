Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is
recommended.

# Environment Set up

* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c
      conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

# Useful commands

```
# Start the application
web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000} --app-dir starter --root-path starter

# Run tests
cd starter
python -m pytest

# Test post to deployed application
python request_live.py

# Create the model, one hot encoder, and label binarizer files
python init.py

# Sanity check
python sanity_check.py # use "tests/test_api.py" as the file to check
```

## Important files/dirs

- starter/slice_output.txt
- starter/screenshots/*

## Know issues

Once the app is deployed on Heroku, some times the following error message arises.

```
failed to pull data from the cloud - Unable to acquire lock. Most likely another DVC process is running or was terminated abruptly.
```

This is a false alert because it downloads the files, in this code it's just a silent warning.

## Repositories

* Create a directory for the project and initialize git and dvc.
    * As you work on the code, continually commit changes. Generated models you want to keep must be committed to dvc.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and
  flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.
* Set up a remote repository for dvc.

# Data

* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.
* Commit this modified data to dvc (we often want to keep the raw data untouched but then can keep updating the cooked
  version).

# Model

* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any
  function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical
      features.
* Write a model card using the provided template.

# API Creation

* Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
        * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the
          column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment

* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail
      before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.
