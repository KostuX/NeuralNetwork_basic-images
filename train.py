from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import os
import getpass
import numpy as np
import tensorflow as tf
import cv2


import keras
from keras.datasets import mnist
from keras.models import Sequential
import keras_metrics
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.layers import RandomFlip, RandomTranslation, RandomZoom, RandomRotation, Input
from keras.optimizers import RMSprop, Adam, SGD
import keras_tuner as kt

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

import src.cfg as cfg
import src.myModels as models_store
import src.helpers as h

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# total_files = sum([len(files) for r, d, files in os.walk(cfg.train_dir)])
h.plotClassBalance()


# h.time_start = 0
h.time_lastCheck = time.time()
h.time_lapse()

momentum = 0.5
learning_rate = 0.001

dropout = 0.2
units = 512

shape = (cfg.img_height, cfg.img_width, cfg.img_channels)

with tf.device('/gpu:0'):
    train_ds, val_ds, test_ds = h.get_ds()
    class_names = train_ds.class_names

    train_f, train_labels = h.split_ds(train_ds)
    test_f, test_labels = h.split_ds(test_ds)

    # train_f = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in train_f])

    # https://medium.com/@ali.oraji/keras-pre-trained-models-for-image-classification-b36c86f8de0d

    pred_model3 = tf.keras.applications.ResNet50V2(
        include_top=False, input_shape=shape, pooling='avg', classes=3, weights='imagenet')
    for layer in pred_model3.layers:
        layer.trainable = False

    # h.tuner(train_ds, overwrite=False)

    ############################
    ####   Define Model   ######
    ############################

    model = Sequential()
    model.add(Input(shape=shape))

    # preprocessing
    model.add(Rescaling(1.0/255))

    # augmentation
    model = models_store.add_augmentation(model)

    # layers
    model = models_store.add_convolution_std(model)

    model.add(Flatten())

    nodes = 256
    for i in range(12):
        model.add(Dense(nodes, activation='relu'))
        if i % 2 == 0:
            model.add(Dropout(0.4))

    # Output layer
    model.add(Dense(cfg.num_classes, activation='softmax'))

    ############################
    ####   Optimisers   ########
    ############################
    optimiser = [
        Adam(),
        SGD(learning_rate=learning_rate,  momentum=momentum),
        RMSprop()
    ]

    ############################
    ####   Compile   ###########
    ############################
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimiser[1],
                  metrics=['accuracy'])

    ############################
    #####   Model Summary   ####
    ############################
    model.summary()

    ############################
    #######   FIT    ###########
    ############################
    printEachEpoch = False

    if cfg.fit:
        history = model.fit(
            train_ds,
            class_weight=h.getClassWeights(train_ds),
            batch_size=cfg.batch_size,
            validation_data=val_ds,
            callbacks=[h.save_callback],
            epochs=cfg.epochs)
        h.plotHistory(history)
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    ############################
    #######   evaluate    ######
    ############################
    # if shuffle=True when creating the dataset, samples will be chosen randomly
    score = model.evaluate(test_ds, batch_size=cfg.batch_size)
    print('Test accuracy:', score[1])

    ############################
    ###   classification    ####
    ############################
    y_pred = model.predict(test_f)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(test_labels, y_pred_classes))

    ############################
    #######   Plot    ##########
    ############################
    test_batch = test_ds.take(1)
    h.plotBatch(test_batch, class_names, model)
