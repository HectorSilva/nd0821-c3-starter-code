import os

import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, recall_score

from starter.constants import CAT_FEATURES
from starter.ml.data import process_data
from starter.train_model import get_artifact


def get_metrics_by_category(model, encoder, lb, options):
    """
    The get_metrics_by_category function takes a model, an encoder, and a label binarizer as inputs.
    It then processes the data using the process_data function to get X and y values for each category.
    The following metrics are calculated: precision recall curve is calculated, f-score and recall score are
    also calculated for each option

    :param model: Specify the model that will be used to make predictions
    :param encoder: Encode the categorical features
    :param lb: Select the label that is used for training
    :param options: Specify the different slices of data that we want to test
    :return: A string with the precision, recall and f-score for each category
    """
    results = ''
    for option in options:
        X, y, _, _ = process_data(df, categorical_features=CAT_FEATURES,
                                  label="salary",
                                  lb=lb,
                                  encoder=encoder,
                                  training=False)
        _y = model.predict(X)

        precision, recall, thresholds = precision_recall_curve(y, _y)
        f1_sc = f1_score(y, _y)
        recall_sc = recall_score(y, _y)
        results += (
                f'**************************************************' +
                f'\nResults for {option}:' +
                f'\n\tPrecision recall curve:\n\t\tprecision {precision[0]} {precision[1]} {precision[2]}' +
                f'\n\t\tRecall {recall[0]} {recall[1]}\n\t\tThresholds {thresholds} ' +
                f'\n\tf1 score {f1_sc}\n\tRecall score {recall_sc}\n'
        )
        with open('slice_output.txt', 'w') as file:
            file.write(results)


if __name__ == "__main__":
    df = pd.read_csv("./data/census_updated.csv")

    marital_status_options = [
        'Divorced',
        'Married-AF-spouse',
        'Married-civ-spouse',
        'Married-spouse-absent',
        'Never-married',
        'Separated',
        'Widowed'
    ]
    abs_path = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(abs_path, 'model')
    model = get_artifact(model_dir, 'trained_model.sav')
    encoder = get_artifact(model_dir, 'onehot_encoder.sav')
    lb = get_artifact(model_dir, 'lb.sav')

    get_metrics_by_category(model=model, encoder=encoder, lb=lb, options=marital_status_options)
