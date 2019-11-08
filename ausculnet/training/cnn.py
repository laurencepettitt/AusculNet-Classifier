#
# def train_cnn():
#     use_class_weighting = True
#     num_epochs = 96
#     num_batch_size = 32
#     optimizer = 'adam'
#     use_focal_loss = True
#     loss = custom_losses.focal_loss if use_focal_loss else losses.categorical_crossentropy
#
#     x = np.expand_dims(x, axis=2)
#     height = len(x[0])
#     num_classes = len(data_set["diagnosis_class"].value_counts())
#     y = data_set['diagnosis_class'].values
#     x_train, x_test, y_train, y_test = prepare_data_set(x, y)
#
#     def model_generator():
#         model = one_dimensional_cnn.construct_model(num_classes, height)
#         model.compile(loss=[loss], metrics=[metrics.categorical_accuracy], optimizer=optimizer)
#         return model
#
#     num_trials = 7
#     training_experiment(model_generator, x_train, x_test, y_train, y_test, num_classes, num_trials, use_class_weighting,
#                         num_epochs, num_batch_size)

