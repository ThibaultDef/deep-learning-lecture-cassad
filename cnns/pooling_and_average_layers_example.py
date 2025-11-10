import cv2

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_sample_images

show_base_image = True
run_pooling_layer = True
run_average_layer = False

# Load some sample of images
sample_dataset = load_sample_images()
print("sample_dataset:", sample_dataset)

images = sample_dataset["images"]

if show_base_image:
    print("image.shape:", images[0].shape)
    cv2.imshow("image", images[0])
    cv2.waitKey(0)


if run_pooling_layer:
    # Pooling parameters
    pool_size = 2

    # Numpy to Tensor
    tf_images = tf.convert_to_tensor(images)
    print("tf_images.shape:", tf_images.shape)

    # We apply a Pooling layer
    pooling_layer = tf.keras.layers.MaxPool2D(pool_size=pool_size)
    sub_sampling = pooling_layer(tf_images)
    print("feature_maps.shape:", sub_sampling.shape)

    # Show the a resulting layer
    np_sub_sampling = sub_sampling.numpy()
    print("np_sub_sampling.shape:", np_sub_sampling.shape)

    if True:
        cv2.imshow("Pooling layer result", np.uint8(np_sub_sampling[0, :, :, :]))
        cv2.waitKey(0)

if run_average_layer:
    # Pooling parameters
    pool_size = 3

    # Numpy to Tensor
    tf_images = tf.convert_to_tensor(images)
    tf_images = tf.keras.layers.Rescaling(scale=1 / 255)(tf_images)
    print("tf_images.shape:", tf_images.shape)

    # We apply a Pooling layer
    pooling_layer = tf.keras.layers.AvgPool2D(pool_size=pool_size)
    sub_sampling = pooling_layer(tf_images)
    print("feature_maps.shape:", sub_sampling.shape)

    # Show the a resulting layer
    np_sub_sampling = sub_sampling.numpy()
    print("np_sub_sampling.shape:", np_sub_sampling.shape)

    if True:
        cv2.imshow("Pooling layer result", np.uint8(255 * np_sub_sampling[0, :, :, :]))
        cv2.waitKey(0)
