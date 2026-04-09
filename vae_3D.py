import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K

class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

def build_3d_vae(input_shape=(64, 64, 64, 1), latent_dim=64):
    # Encoder
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv3D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv3D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv3D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    mu      = layers.Dense(latent_dim, name='mu')(x)
    log_var = layers.Dense(latent_dim, name='log_var')(x)
    z       = Sampling()([mu, log_var])

    encoder = Model(inputs, [mu, log_var, z], name='vae_encoder')

    # VAE loss (reconstruction + KL)
    # Attach via model.add_loss in fusion_model.py
    return encoder, latent_dim