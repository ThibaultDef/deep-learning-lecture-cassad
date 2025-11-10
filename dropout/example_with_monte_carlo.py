import cv2

import numpy as np
import tensorflow as tf


DATA_PATH = "./dropout/data"
MODEL_PATH = "./dropout/models"


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


# Prepare image

image_number = 10

labels = np.load(f"{DATA_PATH}/labels.npy")
image = cv2.imread(f"{DATA_PATH}/image_{image_number}.png", cv2.IMREAD_UNCHANGED)
print("image.shape:", image.shape)
label = labels[image_number]
print("label:", label)

_image = image.reshape(28, 28)
_image = np.expand_dims(_image, axis=0)


# Load models
model2_without_dropout = tf.keras.models.load_model(
    f"{MODEL_PATH}/model_without_dropout.keras"
)
model2_with_dropout = tf.keras.models.load_model(
    f"{MODEL_PATH}/model_with_dropout.keras"
)

# Make predictions
preds_without_dropout = model2_without_dropout.predict(_image).flatten()
index_max = np.argmax(preds_without_dropout)
pred_label_without_dropout = index_max + 1
prob_label_without_dropout = preds_without_dropout[index_max]
print(
    f"Model without dropout predicted image {image_number} as {pred_label_without_dropout} with probability {prob_label_without_dropout}"
)

preds_with_dropout = model2_with_dropout.predict(_image).flatten()
index_max = np.argmax(preds_with_dropout)
pred_label_with_dropout = index_max + 1
prob_label_with_dropout = preds_with_dropout[index_max]
print(
    f"Model with dropout predicted image {image_number} as {pred_label_with_dropout} with probability {prob_label_with_dropout}"
)

# Monte Carlo dropout
if True:
    nb_samples = 200
    y_probas_with_dropout = np.stack(
        [model2_with_dropout(_image, training=True) for _ in range(nb_samples)]
    )
    y_proba_mean_with_dropout = y_probas_with_dropout.mean(axis=0).flatten()
    y_proba_std_with_dropout = y_probas_with_dropout.std(axis=0).flatten()

    index_max = np.argmax(y_proba_mean_with_dropout)
    pred_label_with_mt_estimation = index_max + 1
    prob_mean_label_with_mt_estimation = y_proba_mean_with_dropout[index_max]
    prob_std_label_with_mt_estimation = y_proba_std_with_dropout[index_max]
    print("y_proba_mean_with_dropout:", y_proba_mean_with_dropout)
    print(
        f"""Monte Carlo estimation with {nb_samples} leads to predict image {image_number} as {pred_label_with_mt_estimation}
    with mean probability {prob_mean_label_with_mt_estimation} and standard deviation {prob_std_label_with_mt_estimation}"""
    )
