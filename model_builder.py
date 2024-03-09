import keras
import keras_tuner.src.engine.trial


class ThresholdCallback(keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs["val_accuracy"]
        if val_accuracy >= self.threshold:
            self.model.stop_training = True


class ThresholdCallbackAcc(keras.callbacks.Callback):
    def __init__(self, threshold):
        super(ThresholdCallbackAcc, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs["accuracy"]
        if val_accuracy >= self.threshold:
            self.model.stop_training = True


"""
def epoch_end(trial, model, epoch, logs=None):
    val_accuracy = logs["val_accuracy"]

    if val_accuracy >= 0.9:
        trial.status = keras_tuner.engine.trial.TrialStatus.STOPPED
        model.stop_training = True
        trial.to_proto()
"""


def generate_train_test(raw, split=0.8):
    import numpy as np

    n = int(len(raw)*split)

    training = np.array(raw[:n])
    testing = np.array(raw[n:])

    return training, testing


def generate_tuned_model(ticker_name, name, training, train_targ):
    import keras_tuner as kt
    import numpy as np

    tuner = kt.Hyperband(generate_model,
                         objective=kt.Objective("val_accuracy", "max"),
                         max_epochs=25,
                         factor=3,
                         directory=ticker_name,
                         project_name=name
                         )

    # tuner.on_epoch_end = epoch_end
    # stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    thresh = ThresholdCallback(0.95)

    n = int(0.7*len(training))

    choices = np.random.choice(len(training), n, replace=False)

    train_in = list()
    train_out = list()
    valid_in = list()
    valid_out = list()

    for i in range(len(training)):
        if i not in choices:
            valid_in.append(training[i])
            valid_out.append(train_targ[i])
        else:
            train_in.append(training[i])
            train_out.append(train_targ[i])

    tuner.search(train_in, train_out,
                 epochs=25,
                 validation_data=(valid_in, valid_out),
                 callbacks=[thresh],
                 verbose=1)

    best_hps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hps)

    print("Finding Best Epoch")
    history = model.fit(train_in, train_out,
                        epochs=50,
                        validation_data=(valid_in, valid_out), verbose=1)

    val_epoch = history.history['val_accuracy']
    best_epoch = val_epoch.index(max(val_epoch)) + 1
    print(f"Best epoch: {best_epoch}")

    print("Retraining to best epoch")
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(train_in, train_out,
                   epochs=10,
                   validation_data=(valid_in, valid_out), verbose=0)

    # print("Testing")
    # test_loss, test_acc = hypermodel.evaluate(valid_in, valid_out, verbose=0)
    hypermodel.add(keras.layers.Softmax())

    return hypermodel, [best_hps, best_epoch, 0, 0]


def generate_model(hp):
    import keras

    model = keras.Sequential()
    # model.add(tf.keras.layers.Flatten())

    """
    convo = hp.Boolean(f"ConvolutionLayer", default=True)
    filter_size = hp.Int(f"Convo_Filter_Size", min_value=1, max_value=10)
    kernel_size = hp.Int(f"Convo_Kernel", min_value=1, max_value=10)

    pool = hp.Boolean(f"Pooling_Layer", default=False)
    pool_size = hp.Int(f"Pool_size", min_value=1, max_value=10)

    if convo:
        model.add(tf.keras.layers.Conv1DTranspose(filters=filter_size, kernel_size=kernel_size, activation="relu"))

    if pool:
        model.add(tf.keras.layers.AveragePooling1D(pool_size=pool_size))
    """

    layers = hp.Int("num_layers", min_value=1, max_value=5)
    drop = hp.Boolean(f"drop_out", default=False)
    drop_rate = hp.Float(f"drop_rate", min_value=0, max_value=1, default=0.2)
    units = [hp.Int(f"unit_layer_{i}", min_value=10, max_value=5000) for i in range(layers)]
    lr = hp.Float('learning_rate', max_value=1e-2, min_value=1e-4, sampling='log')

    # model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(units=units[0], activation="relu"))

    for i in range(1, layers):
        model.add(keras.layers.Dense(units=units[i], activation="relu"))

    if drop:
        model.add(keras.layers.Dropout(rate=drop_rate))

    model.add(keras.layers.Dense(3))

    # model.add(keras.layers.Softmax())

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def generate_tuned_func(ticker_name, name, training: dict, train_targ: dict):
    import keras
    import keras_tuner as kt

    tuner = kt.Hyperband(generate_func_model,
                         objective=kt.Objective("val_accuracy", "max"),
                         max_epochs=10,
                         factor=3,
                         directory=ticker_name,
                         project_name=name
                         )

    # tuner.on_epoch_end = epoch_end

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    thresh = ThresholdCallback(0.95)

    tuner.search(training,
                 train_targ,
                 epochs=10,
                 validation_split=0.2,
                 verbose=1,
                 callbacks=[stop_early]
                 )
    best_hps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hps)

    print("Finding best epoch")
    history = model.fit(training,
                        train_targ,
                        epochs=25,
                        validation_split=0.2,
                        callbacks=[thresh],
                        verbose=1)

    val_epoch = history.history['val_accuracy']
    best_epoch = val_epoch.index(max(val_epoch)) + 1

    print("Retraining to best epoch")
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(training,
                   train_targ,
                   epochs=best_epoch,
                   validation_split=0.2,
                   verbose=0)

    return hypermodel


def generate_func_model(hp:  keras_tuner.HyperParameters):
    from keras import layers, Input

    open_input = Input(name="Open", shape=(240,))
    high_input = Input(name="High", shape=(240,))
    low_input = Input(name="Low", shape=(240,))

    open_nodes = hp.Int("open_nodes", 100, 2500)
    open_rate = hp.Float("open_drop", 0, 1)
    open_features = layers.Dense(units=open_nodes, activation="relu")(open_input)
    open_features = layers.Dropout(open_rate)(open_features)

    high_nodes = hp.Int("high_nodes", 100, 2500)
    high_rate = hp.Float("high_drop", 0, 1)
    high_features = layers.Dense(units=high_nodes, activation="relu")(high_input)
    high_features = layers.Dropout(high_rate)(high_features)

    low_nodes = hp.Int("low_nodes", 100, 2500)
    low_rate = hp.Float("low_drop", 0, 1)
    low_features = layers.Dense(units=low_nodes, activation="relu")(low_input)
    low_features = layers.Dropout(low_rate)(low_features)

    concat = layers.concatenate([open_features, high_features, low_features])

    num_layers = 1
    nodes = [hp.Int(f"nodes_{i}", 100, 2500) for i in range(num_layers)]

    first_dense = layers.Dense(nodes[0], activation="relu")(concat)
    # second_dense = layers.Dense(nodes[1], activation="relu")(first_dense)

    logits = layers.Dense(units=3, activation="relu")(first_dense)
    prediction = layers.Softmax()(logits)

    model = keras.Model(inputs=[open_input, high_input, low_input], outputs=[prediction])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    return model


def functional_model():
    import tensorflow as tf

    open_input = tf.keras.Input(name="Open", shape=(720,))
    high_input = tf.keras.Input(name="High", shape=(720,))
    low_input = tf.keras.Input(name="Low", shape=(720,))

    open_features = tf.keras.layers.Dense(units=3000, activation="relu")(open_input)
    open_features = tf.keras.layers.Dropout(0.5)(open_features)

    high_features = tf.keras.layers.Dense(units=3000, activation="relu")(high_input)
    high_features = tf.keras.layers.Dropout(0.5)(high_features)

    low_features = tf.keras.layers.Dense(units=3000, activation="relu")(low_input)
    low_features = tf.keras.layers.Dropout(0.5)(low_features)

    concat = tf.keras.layers.concatenate([open_features, high_features, low_features])

    first_dense = tf.keras.layers.Dense(3000, activation="relu")(concat)
    second_dense = tf.keras.layers.Dense(2000, activation="relu")(first_dense)

    prediction = tf.keras.layers.Dense(units=3, activation="relu")(second_dense)

    model = tf.keras.Model(inputs=[open_input, high_input, low_input], outputs=[prediction])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def generate_tuned_histogram(ticker_name, name, training, train_targ):
    import keras_tuner as kt
    import numpy as np

    tuner = kt.Hyperband(histogram_model,
                         objective=kt.Objective("val_loss", "min"),
                         max_epochs=25,
                         factor=3,
                         directory=ticker_name,
                         project_name=name
                         )

    # thresh = ThresholdCallback(0.95)

    n = int(0.7 * len(training))

    choices = np.random.choice(len(training), n, replace=False)

    train_in = list()
    train_out = list()
    valid_in = list()
    valid_out = list()

    for i in range(len(training)):
        if i not in choices:
            valid_in.append(training[i])
            valid_out.append(train_targ[i])
        else:
            train_in.append(training[i])
            train_out.append(train_targ[i])

    tuner.search(train_in, train_out,
                 epochs=25,
                 validation_data=(valid_in, valid_out),
                 verbose=1
                 )

    best_hps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hps)

    print("Finding Best Epoch")
    history = model.fit(train_in, train_out,
                        epochs=500,
                        validation_data=(valid_in, valid_out), verbose=1)

    val_epoch = history.history['val_loss']
    best_epoch = val_epoch.index(min(val_epoch)) + 1
    print(f"Best epoch: {best_epoch}")

    print("Retraining to best epoch")
    hypermodel = tuner.hypermodel.build(best_hps)
    hypermodel.fit(train_in, train_out,
                   epochs=10,
                   validation_data=(valid_in, valid_out), verbose=0, batch_size=1)

    # print("Testing")
    # test_loss, test_acc = hypermodel.evaluate(valid_in, valid_out, verbose=0)
    hypermodel.add(keras.layers.Softmax())

    return hypermodel, [best_hps, best_epoch, 0, 0]


def histogram_model(hp: keras_tuner.HyperParameters):
    import keras
    model = keras.Sequential()

    # normal = hp.Boolean(f"Normalization_Cond", default=True)

    layers = hp.Int("num_layers", min_value=1, max_value=10)
    drop = hp.Boolean(f"drop_out", default=False)
    drop_rate = hp.Float(f"drop_rate", min_value=0, max_value=0.8, default=0.2)
    units = [hp.Int(f"unit_layer_{i}", min_value=10, max_value=5000) for i in range(layers)]
    lr = hp.Float('learning_rate', max_value=1e-2, min_value=1e-4, sampling='log')

    model.add(keras.layers.Flatten())

    # if normal:
    #    model.add(keras.layers.BatchNormalization())

    for i in range(layers):
        model.add(keras.layers.Dense(units=units[i], activation="relu"))

    if drop:
        model.add(keras.layers.Dropout(rate=drop_rate))

    model.add(keras.layers.Dense(168))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=[keras.metrics.MeanAbsoluteError()]
    )

    return model


def toy_histogram_model():
    import keras
    model = keras.Sequential()

    layers = 2
    drop = True
    drop_rate = 0.2
    units = [100, 100]
    lr = 1e-3

    model.add(keras.layers.Flatten())

    for i in range(layers):
        model.add(keras.layers.Dense(units=units[i], activation="relu"))

    if drop:
        model.add(keras.layers.Dropout(rate=drop_rate))

    model.add(keras.layers.Dense(200))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.MeanSquaredError(),
        metrics=["accuracy"]
    )

    return model
