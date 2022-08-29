# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model created to predict the salary of a person according to
some features or categorical values
(income, outcome, salary, gender, country, etc.)

Model date: 2022-08-28
Developed by Udacity

## Intended Use

Use cases for this model:

- Predict the salary of a person given:
    - Age
    - Work class
    - Education
    - Education number
    - Marital status
    - Occupation
    - Relationship
    - Race
    - Sex
    - Capital gain
    - Capital loss
    - Hours per week
    - Native Country
    - Salary
- Take decisions according the prediction of the salary of a person
- Give or deny benefits/services according the salary predicted

## Training Data

Data sets used:

- census.csv
  This dataset includes the census results from a huge amount of people,
  this includes data necessary to predict with high accuracy the salary of any person.

## Evaluation Data

Data sets used:

- census.csv
- Tests to measure the bias of the model
- Tests to check the accuracy of the model

## Metrics

Metrics used for this model:

- Precision recall curve
- Recall score
- f1 score

Each of them are calculated by a given option in a categorical value

This is a sample output of the described metrics:

```
Precision recall curve:
    precision 0.2408095574460244 0.7872527472527473 1.0
    Recall 1.0 0.6852442290524168
    Thresholds [0 1] 
f1 score 0.7327151234147007
Recall score 0.6852442290524168
```

## Ethical Considerations

In order to use this model, there should be written consent of each of the people who sent their data for privacy
considerations

## Caveats and Recommendations

The function to create metrics based on the categorical values
in this model only works on the marital status, it's desirable to add more metrics for all the other categorical values
and options.
