"""
Creates all the basic files:
- Trained model
- OneHot encoder
- Label Binarizer

and saves them in the defined directory (model)
"""
import os

import starter.train_model as tm

if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(abs_path, 'model')
    model_filename = 'trained_model.sav'
    onehot_encoder_filename = 'onehot_encoder.sav'
    lb_filename = 'lb.sav'

    # Get the data
    data = tm.get_data()

    # Split the data and the process it
    X_train, y_train, encoder, lb, test = tm.train_split(data)
    test_X_train, test_y_train, test_encoder, test_lb = tm.process_test_data(test, encoder, lb)

    # Train the model
    model = tm.train_ml_model(test_X_train, test_y_train)

    # Save all the artifacts in the model directory
    tm.save_artifact(model, model_dir, model_filename)
    tm.save_artifact(encoder, model_dir, onehot_encoder_filename)
    tm.save_artifact(lb, model_dir, lb_filename)
