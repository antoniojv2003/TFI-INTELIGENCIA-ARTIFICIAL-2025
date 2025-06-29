import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Input, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from config import IMAGE_SIZE, LEARNING_RATE

def build_net(input_shape, num_classes):
    """Construye un modelo personalizado con componentes vistos en clase."""
    inputs = Input(shape=input_shape)
    
    # Convolución 1
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Capa final
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_model(num_classes):
    """Construye y compila el modelo personalizado."""
    model = build_net(input_shape=(*IMAGE_SIZE, 3), num_classes=num_classes)
        
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model 