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
    # Plain Conv->ReLU->Pool (no BatchNorm: it was unstable on this 13k-class
    # dataset with a Python generator and small per-class sample count).
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    return x


def build_model_b(num_classes, img_size=128):
    """Model B - deep CNN (plain, no BatchNorm), 4 blocks, Flatten head."""
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = _conv_block(inputs, 32)
    x = _conv_block(x, 64)
    x = _conv_block(x, 128)
    x = _conv_block(x, 256)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = _finalize(x, num_classes)
    return keras.Model(inputs, outputs, name='model_b_deep_cnn')


def build_model_c(num_classes, img_size=128, weights='imagenet'):
    """Model C - MLP baseline (no convolutions).

    weights accepted but unused (shared get_model signature); starts from scratch.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = layers.Flatten()(inputs)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = _finalize(x, num_classes)
    return keras.Model(inputs, outputs, name='model_c_mlp')


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
