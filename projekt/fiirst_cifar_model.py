import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

import tempfile
from os import path
from tensorflow.keras import datasets

# Model / data parameters
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)
MODEL_PATH = './cifar_models/first_cifar_model_30_epochs.h5'

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

if path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Compile
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print('Loaded from: ' + MODEL_PATH)
else:
    model = make_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    # Train the digit classification model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    checkpoint_filepath = './tmp/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.fit(
        train_images,
        train_labels,
        epochs=30,
        validation_split=0.1,
        batch_size=64,
        callbacks=[model_checkpoint_callback],
        verbose=2
    )
    
    # save model
    model.save(MODEL_PATH)
    print('Saved to: ' + MODEL_PATH)
    
    
model.evaluate(test_images, test_labels)