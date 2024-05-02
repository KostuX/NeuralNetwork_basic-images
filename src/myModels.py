import src.cfg as cfg
import tensorflow as tf

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.layers import RandomFlip,RandomTranslation,RandomZoom,RandomRotation


def model_basic():
        return tf.keras.models.Sequential([
        Rescaling(1.0/255),
        Conv2D(16, (3,3), activation = 'relu', input_shape = (cfg.img_height, cfg.img_width, cfg.img_channels)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation = 'relu'),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation = 'relu'),
        MaxPooling2D(2,2),
        Flatten(), # flatten multidimensional outputs into single dimension for input to dense fully connected layers
        Dense(512, activation = 'relu'),
        Dropout(0.2),
        Dense(cfg.num_classes, activation = 'softmax')
    ])


def model_pred_1():
    pred_model = tf.keras.applications.VGG16(include_top=False, input_shape = (cfg.img_height, cfg.img_width, cfg.img_channels), pooling='avg', classes=3, weights='imagenet')
    for layer in pred_model.layers:
        layer.trainable=False

    model = tf.keras.models.Sequential([
        pred_model,        
        Dense(cfg.num_classes, activation = 'softmax')
    ])
    return model


def add_augmentation(model):
    # Augmentation  layer    
    #model.add(RandomFlip("horizontal and vertical"))
    model.add(RandomRotation(cfg.random_Rotation))
    model.add(RandomTranslation(height_factor=cfg.translation_factors, width_factor=cfg.translation_factors))
    model.add(RandomZoom(cfg.randomZoom)) 
    return model


def add_convolution_std(model):
    # conv layer
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))    
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2))    
    model.add(Conv2D(64, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(2,2)) 

    return model

    
