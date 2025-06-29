import pandas as pd
import numpy as np
import tensorflow as tf
import os
from config import COMPETITION_TASKS, IMAGE_SIZE


def load_and_preprocess_data(csv_path):
    """Carga y preprocesa el conjunto de datos CheXpert.

    Estrategia híbrida de incertidumbre:
        - U‑Ones para Atelectasis y Edema  (‑1 → 1)
        - U‑Ignore para el resto          (‑1 → NaN, luego NaN → 0)
    """
    df = pd.read_csv(csv_path)

    for col in COMPETITION_TASKS:
        if col in ['Atelectasis', 'Edema']:
            # U‑Ones: tratamos la incertidumbre como positiva
            df[col] = df[col].fillna(0).replace(-1, 1)
        else:
            # U‑Ignore: marcamos la incertidumbre como NaN y la ignoramos
            df[col] = df[col].replace(-1, np.nan)
            df[col] = df[col].fillna(0)

    return df


def create_dataset(df, batch_size, shuffle=True):
    """Crea un dataset de TensorFlow."""

    def preprocess_image(image_path, label):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        # Normalizar a [0, 1]
        img = img / 255.0
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices(
        (df['Path'].values, df[COMPETITION_TASKS].values.astype(np.float32))
    )
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
