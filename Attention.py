import tensorflow as tf
from tensorflow.keras import layers, Model

def build_se_attention(input_shape=(64, 64, 64, 1), reduction=16):
    """
    Squeeze-and-Excitation network that learns to weight
    brain regions (hippocampus, entorhinal cortex) automatically.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv3D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling3D(2)(x)
    x = layers.Conv3D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling3D(2)(x)

    # SE channel attention
    se = layers.GlobalAveragePooling3D()(x)
    se = layers.Dense(128 // reduction, activation='relu')(se)
    se = layers.Dense(128, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 1, 128))(se)
    x  = layers.Multiply()([x, se])

    x = layers.GlobalAveragePooling3D()(x)
    features = layers.Dense(128, activation='relu')(x)

    return Model(inputs, features, name='se_attention')