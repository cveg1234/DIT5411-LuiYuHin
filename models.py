"""Three CNN architectures to compare, as required by the assignment.

Every model ends with Dense(num_classes, softmax) -> 13,065 neurons.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _finalize(x, num_classes):
    # float32 output keeps softmax numerically stable under mixed precision
    x = layers.Dense(num_classes, activation='softmax', name='output',
                     dtype='float32')(x)
    return x


def build_model_a(num_classes, img_size=128):
    """Model A - simple baseline CNN."""
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = _finalize(x, num_classes)
    return keras.Model(inputs, outputs, name='model_a_simple_cnn')


def _conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    return x


def build_model_b(num_classes, img_size=128):
    """Model B - deeper CNN with batch normalisation and dropout."""
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = _conv_block(inputs, 32)
    x = layers.Dropout(0.15)(x)
    x = _conv_block(x, 64)
    x = layers.Dropout(0.20)(x)
    x = _conv_block(x, 128)
    x = layers.Dropout(0.25)(x)
    x = _conv_block(x, 256)
    x = layers.Dropout(0.30)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.40)(x)
    outputs = _finalize(x, num_classes)
    return keras.Model(inputs, outputs, name='model_b_deep_cnn')


def build_model_c(num_classes, img_size=128, weights='imagenet'):
    """Model C - MobileNetV2 transfer learning."""
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=weights,
        pooling=None,
    )
    base.trainable = False
    inputs = base.input
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = _finalize(x, num_classes)
    return keras.Model(inputs, outputs, name='model_c_mobilenetv2')


MODEL_BUILDERS = {
    'A': build_model_a,
    'B': build_model_b,
    'C': build_model_c,
}


def get_model(name, num_classes, img_size=128, weights='imagenet'):
    name = name.upper()
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model {name}. Choose from {list(MODEL_BUILDERS)}")
    if name == 'C':
        return MODEL_BUILDERS[name](num_classes, img_size, weights=weights)
    return MODEL_BUILDERS[name](num_classes, img_size)
