import cv2

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_sample_images

show_base_image = True
run_filter_maps = True
show_filter_map_result = True

# Load some sample of images
sample_dataset = load_sample_images()
print("sample_dataset:", sample_dataset)

images = sample_dataset["images"]

if show_base_image:
    print("image.shape:", images[0].shape)
    cv2.imshow("image", images[0])
    cv2.waitKey(0)


if run_filter_maps:
    # We apply some transformations, specially here a reshaping and rescaling
    tf_images = tf.keras.layers.CenterCrop(height=70, width=120)(images)
    tf_images = tf.keras.layers.Rescaling(scale=1 / 255)(tf_images)
    print("tf_images.shape:", tf_images.shape)

    # We apply a Convolution layer with 32 filters
    conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
    feature_maps = conv_layer(tf_images)
    print("feature_maps.shape:", feature_maps.shape)

    # Show the a resulting layer
    np_feature_maps = feature_maps.numpy()
    np_feature_maps_example = np.int8(255 * np_feature_maps[0, :, :, -1]).reshape(
        64, 114
    )
    print("np_feature_maps_example.shape:", np_feature_maps_example.shape)

    if show_filter_map_result:
        cv2.imshow("feature map result", np_feature_maps_example)
        cv2.waitKey(0)
