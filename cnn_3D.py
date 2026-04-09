import tensorflow as tf
from tensorflow.keras import layers, Model

def build_3d_cnn_encoder(input_shape=(64, 64, 64, 1), feature_dim=256):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Conv3D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)

    x = layers.Conv3D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)

    x = layers.Conv3D(128, 3, padding='same', activation='relu', name='last_conv3d')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    features = layers.Dense(feature_dim, activation='relu')(x)

    return Model(inputs, features, name='cnn3d_encoder')