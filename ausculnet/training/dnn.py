import respiratory_sounds
from keras import metrics, losses
from sklearn.model_selection import train_test_split

import ausculnet.losses as custom_losses
from ausculnet.models import simple_dnn
from ausculnet.preprocessing import extract_features
from ausculnet.preprocessing.data_augmentation import augment_samples_by_random_crop
from ausculnet.preprocessing.resampling import downsample_diagnosis_class
from ausculnet.preprocessing.tools import one_hot_encode, filter_samples_by_classes
from ausculnet.training.train import training_experiment


def train_dnn():
    optimizer = 'adam'
    use_focal_loss = False
    loss = custom_losses.focal_loss if use_focal_loss else losses.categorical_crossentropy

    num_all_classes = respiratory_sounds.num_classes()
    all_classes = respiratory_sounds.get_diagnosis_classes()
    unwanted_classes = [1, 5]
    wanted_classes = [c for c in all_classes if c not in unwanted_classes]
    num_classes = len(wanted_classes)

    features = ["mfccs", "chroma", "mel", "contrast"]  # , "tonnetz"]

    data_set = respiratory_sounds.get_recordings_patients_join()

    data_set = downsample_diagnosis_class(data_set, 4, 100)

    data_set = filter_samples_by_classes(data_set=data_set, classes=wanted_classes)

    train_data_set, test_data_set, _, _ = train_test_split(data_set, data_set['diagnosis_class'], test_size=0.3,
                                                           random_state=42)

    train_data_set = augment_samples_by_random_crop(train_data_set, frac=0.5, random_state=42)
    train_data_set = downsample_diagnosis_class(train_data_set, 4, 90)

    x_train = extract_features.get_features_stacked(data_set=train_data_set, features=features)
    x_test = extract_features.get_features_stacked(data_set=test_data_set, features=features)
    y_train = one_hot_encode(train_data_set["diagnosis_class"].tolist())
    y_test = one_hot_encode(test_data_set["diagnosis_class"].tolist())

    inputs_and_targets = x_train, x_test, y_train, y_test

    def compiled_model_generator():
        model = simple_dnn.construct_model(num_classes=num_classes)
        model.compile(loss=[loss], metrics=[metrics.categorical_accuracy], optimizer=optimizer)
        return model

    use_class_weighting = True
    num_epochs = 96
    num_batch_size = 32

    # return (
    #     compiled_model_generator,
    #     inputs_and_targets,
    #     num_classes,
    #     use_class_weighting,
    #     num_epochs,
    #     num_batch_size
    # )

    # TODO - //
    #  training_experminet.build(compiled_model_generator,
    #         x_train,
    #         x_test,
    #         y_train,
    #         y_test,
    #         num_classes,
    #         use_class_weighting,
    #         num_epochs,
    #         num_batch_size)
    #  training_experiment.run(num_trials)

    num_trials = 7
    training_experiment(
        compiled_model_generator,
        inputs_and_targets,
        num_classes,
        num_trials,
        use_class_weighting,
        num_epochs,
        num_batch_size
    )


if __name__ == '__main__':
    train_dnn()
