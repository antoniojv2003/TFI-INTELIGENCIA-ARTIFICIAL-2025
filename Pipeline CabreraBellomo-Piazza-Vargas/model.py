import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, SeparableConv2D, BatchNormalization, ReLU,
    GlobalAveragePooling2D, Dense, Input, MaxPooling2D, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from config import IMAGE_SIZE, LEARNING_RATE


def conv_block(x, filters, kernel_size=3, strides=1, use_pooling=False):
    x = SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    if use_pooling:
        x = MaxPooling2D(pool_size=(2, 2))(x)
    return x


def build_net(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Bloque 1
    x = conv_block(inputs, 32)
    x = conv_block(x, 32, use_pooling=True)

    # Bloque 2
    x = conv_block(x, 64)
    x = conv_block(x, 64, use_pooling=True)

    # Bloque 3
    x = conv_block(x, 128)
    x = conv_block(x, 128, use_pooling=True)

    # Bloque 4
    x = conv_block(x, 256)
    x = conv_block(x, 256)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # Capa densa intermedia opcional
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)

    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model


def build_model(num_classes):
    model = build_net(input_shape=(*IMAGE_SIZE, 3), num_classes=num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model
