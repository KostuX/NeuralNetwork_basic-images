import tensorflow as tf
from datetime import datetime, timedelta
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import src.cfg as cfg
from IPython.display import clear_output
import keras_tuner as kt
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.layers import RandomFlip, RandomTranslation, RandomZoom, RandomRotation, Input

from keras.models import Sequential
time_start = 0
time_lastCheck = time.time()


# earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
save_callback = tf.keras.callbacks.ModelCheckpoint(
    "pneumonia.keras", save_freq='epoch', save_best_only=True)


def time_lapse():
    global time_start, time_lastCheck
    time_curr = time.time()
    if time_start == 0:
        time_start = time_curr

    time_fromStart = round(time_curr - time_start)
    time_fromLast = round(time_curr - time_lastCheck)
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f'\nTime: {formatted_time} \nStarted: {timedelta(seconds=time_fromStart)} ago\nLast checkpoint: {timedelta(seconds=time_fromLast)} ago')
    time_lastCheck = time_curr


def get_ds():
    # create training,validation and test datatsets
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        cfg.train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(cfg.img_height, cfg.img_width),
        batch_size=cfg.batch_size,
        labels='inferred',
        shuffle=True)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        cfg.test_dir,
        seed=None,
        image_size=(cfg.img_height, cfg.img_width),
        batch_size=cfg.batch_size,
        labels='inferred',
        shuffle=True)
    return train_ds, val_ds, test_ds


def split_ds(ds):
    train_features_only = []
    train_labels_only = []
    for batch_data, batch_labels in ds:
        train_features_only.append(batch_data)
        train_labels_only.append(batch_labels)
    train_features_only = tf.concat(train_features_only, axis=0)
    train_labels_only = tf.concat(train_labels_only, axis=0)

    train_features_only = train_features_only.numpy()
    train_labels_only = train_labels_only.numpy()
    return train_features_only, train_labels_only


def printPreview():
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()


class epoch_callback(tf.keras.callbacks.Callback):

    def __init__(self):
        self.val_acc = []
        self.acc = []

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)

        _val_acc = logs.get('val_accuracy')
        _acc = logs.get('accuracy')

        self.val_acc.append(_val_acc)
        self.acc.append(_acc)
        maximum = max(self.val_acc)

        time_lapse()

        plt.plot(self.acc)
        plt.plot(self.val_acc)

        plt.title(f'Model Accuracy. Epoch: {epoch}')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if maximum > _val_acc:
            plt.axhline(y=maximum, color='r', linewidth=0.5, linestyle='--')
        plt.legend([f'Train: {round(_acc*100,1)}%', f'Val: {round(_val_acc*100,1)}%',
                   f'Max: {round(maximum*100,1)}%'], loc='upper left')

        plt.show()


def getClassWeights(ds):
    labels = np.asarray(list(ds.unbatch().map(lambda x, y: y)))
    count_0 = np.sum(labels == 0)
    count_1 = np.sum(labels == 1)
    count_2 = np.sum(labels == 2)

    max_class = max(count_0, count_1, count_2)

    weight_0 = max_class/count_0
    weight_1 = max_class/count_1
    weight_2 = max_class/count_2
    class_wght = {0: weight_0, 1: weight_1, 2:  weight_2}
    print(f'Weights: {class_wght}')
    return class_wght


def plotClassBalance():
    dirs = [name for name in os.listdir(cfg.train_dir) if os.path.isdir(
        os.path.join(cfg.train_dir, name))]
# [BACTERIAL, NORMAL, VIRA]
    class_size = {}
    for classes_ in dirs:
        x = sum([len(files)
                for r, d, files in os.walk(f'{cfg.train_dir}/{classes_}')])
        class_size[classes_] = x
    labels = list(class_size.keys())
    values = list(class_size.values())
    plt.bar(labels, values)
    plt.title(
        f'{labels[0]} : {values[0]}\n {labels[1]} : {values[1]}\n {labels[2]} : {values[2]}')

    plt.show()
    print(list(class_size.keys()), list(class_size.values()))


def plotHistory(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')


def plotBatch(test_batch, class_names, model):
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            # perform a prediction on this image
            prediction = model.predict(tf.expand_dims(images[i].numpy(), 0))
            plt.title('Actual:' + class_names[labels[i].numpy()] + '\nPredicted:{} {:.2f}%'.format(
                class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.show()


def tuner(ds, overwrite=False):
    ############################
    ####   tunner   ############
    ############################
    train_f, train_labels = split_ds(ds)

    def model_builder(hp):
        model = Sequential()

        model.add(Input(shape=(cfg.img_height, cfg.img_width, cfg.img_channels)))

        model.add(RandomFlip("horizontal and vertical"))

        rotation = hp.Choice('Random Rotation', values=[0.3, 0.5, 0.7])
        model.add(RandomRotation(rotation))

        factors = hp.Choice('Translation factors', values=[0.3, 0.5, 0.7])
        model.add(RandomTranslation(
            height_factor=factors, width_factor=factors))

        zoom = hp.Choice('RandomZoom', values=[0.3, 0.5, 0.7])
        model.add(RandomZoom(zoom))

        model.add(Rescaling(1.0/255))
        model.add(Conv2D(16, (3, 3), activation='relu', ))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(keras.layers.Flatten())

        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units,
                  activation='relu', name='denses'))

        dropout = hp.Choice('Dropout', values=[0.1, 0.2, 0.4, 0.6])
        model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(cfg.num_classes, activation='softmax'))

        hp_learning_rate = hp.Choice(
            'learning_rate', values=[0.01, 0.001, 0.0001])
        momentum = hp.Choice('momentum', values=[0.3, 0.5, 0.7])
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=hp_learning_rate,  momentum=momentum),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        return model

    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=30,
                         factor=3,
                         directory='assignment2',
                         project_name='B00148740',
                         overwrite=overwrite
                         )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)

    tuner.search(train_f, train_labels, epochs=30,
                 validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    units = best_hps.get('units')
    rate = best_hps.get('learning_rate')

    print(f"""
            The hyperparameter search is complete.
            Learning rate:  {best_hps.get('learning_rate')},
            Momentum: {best_hps.get('momentum')},
            Units: {best_hps.get('units')} ,
            Random Rotation:  {best_hps.get('Random Rotation')},
            Translation factors: {best_hps.get('Translation factors')},
            RandomZoom: {best_hps.get('RandomZoom')},
            Dropout:  {best_hps.get('Dropout')},
            """)

    return units, rate
