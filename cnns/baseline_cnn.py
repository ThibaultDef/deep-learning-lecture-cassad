import os

from functools import partial

import cv2

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


MAIN_PATH = "./cnns"
EPOCHS = 3
LR = 0.001

run_model_training = False
show_feature_map = True

subfolders = ["data", "models", "figures"]
for subfolder in subfolders:
    if not os.path.exists(f"{MAIN_PATH}/{subfolder}"):
        os.makedirs(f"{MAIN_PATH}/{subfolder}")


def plot_losses(nb_epochs, loss, val_loss, path_to_save):
    epochs = range(1, nb_epochs + 1)

    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.savefig(path_to_save)


def save_test_images(ds_test, path_to_save):
    for images, labels in ds_test:
        _images = images.numpy()
        _labels = labels.numpy()

        np.save(f"{path_to_save}/labels.npy", _labels)

        for i in range(_images.shape[0]):
            _image = _images[i, :, :].reshape(28, 28)
            cv2.imwrite(f"{path_to_save}/image_{i}.png", _image)

        break


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.squeeze(tf.cast(image, tf.float32), axis=-1) / 255.0, label


(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

if show_feature_map:
    save_test_images(
        ds_test=ds_test.batch(32).prefetch(tf.data.AUTOTUNE),
        path_to_save=f"./{MAIN_PATH}/data",
    )


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(32).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(32).prefetch(tf.data.AUTOTUNE)

if run_model_training:
    DefaultConv2D = partial(
        tf.keras.layers.Conv2D, kernel_size=3, padding="same", activation="relu"
    )

    """
    model = tf.keras.Sequential(
        [
            DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
            tf.keras.layers.MaxPool2D(),
            DefaultConv2D(filters=128),
            DefaultConv2D(filters=128),
            tf.keras.layers.MaxPool2D(),
            DefaultConv2D(filters=256),
            DefaultConv2D(filters=256),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation="relu"),
            tf.keras.layers.Dense(units=64, activation="relu"),
            tf.keras.layers.Dense(units=10, activation="softmax"),
        ]
    )
    """

    model = tf.keras.Sequential(
        [
            DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
            tf.keras.layers.MaxPool2D(),
            DefaultConv2D(filters=50),
            DefaultConv2D(filters=50),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=20, activation="relu"),
            tf.keras.layers.Dense(units=10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(ds_train, validation_data=ds_test, epochs=EPOCHS)

    model.save(f"./{MAIN_PATH}/models/baseline_cnn.keras")

    losses_training_without_dropout = history.history["loss"]
    losses_validation_without_dropout = history.history["val_loss"]

    plot_losses(
        nb_epochs=EPOCHS,
        loss=losses_training_without_dropout,
        val_loss=losses_validation_without_dropout,
        path_to_save=f"{MAIN_PATH}/figures/loss.png",
    )


if show_feature_map:
    loaded_model = tf.keras.models.load_model(
        f"./{MAIN_PATH}/models/baseline_cnn.keras"
    )
    loaded_model.summary()

    for image, label in ds_test:
        break

    print("layers:", loaded_model.layers)

    image = image.numpy()[0, :, :]
    x = np.expand_dims(image, axis=-1)
    x = np.expand_dims(x, axis=0)
    for layer in loaded_model.layers:
        x = layer(x)
        if layer.name == "conv2d":
            break

    x = np.squeeze(x, axis=0)
    x = x[:, :, 0]
    x = np.int8(255 * x)

    cv2.imshow("original_image", np.int8(255 * image))
    cv2.waitKey(0)

    cv2.imshow("first_conv2d_output", x)
    cv2.waitKey(0)
