import tensorflow as tf
from tensorflow.keras import layers, Model
from models.cnn_3D import build_3d_cnn_encoder
from models.vae_3D import build_3d_vae
from models.Attention import build_se_attention

NUM_CLASSES = 4

def build_full_model(input_shape=(64, 64, 64, 1)):
    inputs = tf.keras.Input(shape=input_shape, name='mri_input')

    # Branch 1: 3D CNN
    cnn_enc     = build_3d_cnn_encoder(input_shape)
    cnn_feat    = cnn_enc(inputs)           # (batch, 256)

    # Branch 2: VAE (use only mu as deterministic feature at inference)
    vae_enc, _  = build_3d_vae(input_shape, latent_dim=64)
    mu, log_var, z = vae_enc(inputs)        # (batch, 64)

    # Branch 3: SE attention
    attn_enc    = build_se_attention(input_shape)
    attn_feat   = attn_enc(inputs)          # (batch, 128)

    # Fusion
    fused = layers.Concatenate()([cnn_feat, mu, attn_feat])  # (batch, 448)
    fused = layers.Dense(256, activation='relu')(fused)
    fused = layers.Dropout(0.4)(fused)
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.Dropout(0.3)(fused)
    output = layers.Dense(NUM_CLASSES, activation='softmax', name='prediction')(fused)

    model = Model(inputs, output, name='alzheimer_fusion')

    # KL loss for VAE regularization
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    model.add_loss(0.001 * kl_loss)   # weight = β, tune this

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model