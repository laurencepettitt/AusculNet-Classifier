from datetime import datetime

import numpy as np
from keras.callbacks import TensorBoard
from ausculnet.analysis import dataset_analysis
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix


def predict_from_model(model, x, y, num_batch_size):
    """
    Given model, makes prediction and returns tuple of (true classes, predicted classes), each a list of ints.

    Args:
        model: trained keras model to use to make prediction
        x: input to make prediction from
        y: expected output in one-hot encoded form
        num_batch_size: batch_size for batching predictions

    Returns: tuple of lists of ints
        (true classes, predicted classes)
    """
    y_true_classes = np.argmax(y, axis=1)
    y_pred_classes = model.predict_classes(x, batch_size=num_batch_size)
    return y_true_classes, y_pred_classes


def print_classification_report(y_true_classes, y_pred_classes, report_name):
    """
    Print text report showing main classification metrics.

    Args:
        y_true_classes: list of ints
        y_pred_classes: list of ints
        report_name: str

    Returns:
        None
    """
    print("Classification Report: ", report_name)
    print(classification_report(y_true_classes, y_pred_classes))


def prediction_evaluation(model, x, y, num_classes, num_batch_size, report_name):
    """
    Compute precision, recall, F-measure and support for each class

    Args:
        model: trained keras model
        x: input to model
        y: expected output to model
        num_classes: how many classes model was trained to predict for
        num_batch_size: batch size for running prediction over inputs
        report_name: str report name

    Returns: tuple
        precision, recall, F-measure, support for each class
    """
    y_true, y_pred = predict_from_model(model, x, y, num_batch_size)
    return precision_recall_fscore_support(y_true, y_pred, labels=tuple(range(num_classes)))


def print_confusion_matrix(model, x, y, num_batch_size):
    """
    Prints confusion matrix after making prediction on model

    Args:
        model: keras model
        x: input to model
        y: expected output
        num_batch_size: batch size for running prediction over inputs

    Returns:

    """
    y_true, y_pred = predict_from_model(model, x, y, num_batch_size)
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print('Confusion matrix:\n', conf_mat)


def training_experiment(compiled_model_generator,
                        data,
                        num_classes,
                        num_trials,
                        use_class_weighting,
                        num_epochs,
                        num_batch_size):
    """
    Run a training experiment.

    Training experiment involves one or more trials.
    Each trial gets a (new, untrained) model from compiled_model_generator()
    and trains the model using the supplied hyper-parameters.
    The result is an average of the results over all the trials.

    Args:
        compiled_model_generator: generates a new, untrained, compiled keras model
        data: tuple of (x_train, x_test, y_train, y_test)
        num_classes: number of classes in output of model
        num_trials: num of times to repeat training
        use_class_weighting: boolean, true to use class weighting
        num_epochs: int, num epochs in training
        num_batch_size: int, batch size in training

    Returns:

    """
    x_train, x_test, y_train, y_test = data
    class_distribution = dataset_analysis.get_num_recordings_per_diagnosis_class(relative=True)
    if use_class_weighting:
        class_weight = {(class_key, (1 / num_classes) / dist) for class_key, dist in class_distribution.items()}
    else:
        class_weight = None

    evaluations = []
    for i in range(num_trials):
        model = compiled_model_generator()

        # *** LOGGING *** #
        tensor_board = TensorBoard(log_dir="logs/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))

        # *** PRE-TRAINING EVALUATION ** #
        score = model.evaluate(x_test, y_test, verbose=1)
        accuracy = score[1]
        print("Pre-training accuracy: %.4f%%" % (100 * accuracy))

        # *** FIT TO MODEL *** #
        start = datetime.now()

        model.fit(
            x_train,
            y_train,
            batch_size=num_batch_size,
            epochs=num_epochs,
            validation_data=(x_test, y_test),
            class_weight=class_weight,
            callbacks=[tensor_board]
        )

        duration = datetime.now() - start
        print("Training completed in time: ", duration)

        print(model.summary())

        # *** POST-TRAINING EVALUATION *** #
        set_evaluations = []
        for x, y, name in [(x_train, y_train, "train"), (x_test, y_test, "test")]:
            set_evaluations.append(
                prediction_evaluation(model, x, y, num_classes, num_batch_size, report_name=name)
            )
            print_confusion_matrix(model, x, y, num_batch_size)
        evaluations.append(set_evaluations)

    average_evaluations = np.mean(evaluations, axis=0)
    for set_evaluation in average_evaluations:
        print()
        for evaluation in set_evaluation:
            print(",".join("%.2f" % x for x in evaluation))

    return evaluations, average_evaluations
