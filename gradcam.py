import tensorflow as tf
import numpy as np
import cv2


# ---------------------------
# AUTO FIND LAST CONV3D LAYER
# ---------------------------
def get_last_conv3d_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv3D):
            return layer.name
    raise ValueError("❌ No Conv3D layer found in model")


# ---------------------------
# GENERATE 3D GRAD-CAM (Sequential-safe)
# ---------------------------
def generate_3d_gradcam(model, volume, class_index=None):
    """
    Works with Sequential models that have no explicit Input layer.
    volume: np.ndarray of shape (1, 64, 64, 64, 1)
    """

    volume_tensor = tf.cast(volume, tf.float32)

    # ✅ Warm-up call to force the model to build all layer weights
    _ = model(volume_tensor, training=False)

    last_conv_layer_name = get_last_conv3d_layer(model)

    # ✅ Split model into two parts manually
    # Part 1: input → last conv layer output
    # Part 2: everything after the last conv layer
    found = False
    layers_before = []   # up to and including last conv
    layers_after  = []   # everything after last conv

    for layer in model.layers:
        if not found:
            layers_before.append(layer)
            if layer.name == last_conv_layer_name:
                found = True
        else:
            layers_after.append(layer)

    # ✅ Forward pass through part 1, tracking gradients on conv output
    x = volume_tensor
    for layer in layers_before:
        x = layer(x, training=False)

    conv_output = x  # shape: (1, D, H, W, filters)

    # ✅ Forward pass through part 2 with GradientTape
    with tf.GradientTape() as tape:
        tape.watch(conv_output)

        out = conv_output
        for layer in layers_after:
            out = layer(out, training=False)

        predictions = out  # shape: (1, num_classes)

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))

        loss = predictions[:, class_index]

    # ✅ Gradient of class score w.r.t. conv output
    grads = tape.gradient(loss, conv_output)

    if grads is None:
        raise ValueError(
            "❌ Gradients are None. "
            "Make sure your model has Conv3D layers and is not fully quantized."
        )

    # Global average pool over spatial dims → shape: (filters,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    # Weight each feature map → shape: (D, H, W)
    conv_output_squeezed = conv_output[0]                        # (D, H, W, filters)
    heatmap = tf.reduce_sum(
        conv_output_squeezed * pooled_grads, axis=-1
    )                                                            # (D, H, W)

    # ReLU + normalize
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()   # shape: (D, H, W)


# ---------------------------
# OVERLAY ON 3 PLANES
# ---------------------------
def overlay_3d_on_slices(volume, heatmap):
    """
    Returns axial, coronal, sagittal BGR overlays as numpy arrays.
    volume:  (1, D, H, W, 1)
    heatmap: (d, h, w)  — may differ in size from volume
    """

    vol = volume[0, :, :, :, 0]   # (D, H, W)

    D, H, W = vol.shape
    OUTPUT_SIZE = 256

    mid_vol  = [D // 2, H // 2, W // 2]
    mid_heat = [s // 2 for s in heatmap.shape]

    slices = [
        vol[mid_vol[0], :, :],    # axial
        vol[:, mid_vol[1], :],    # coronal
        vol[:, :, mid_vol[2]],    # sagittal
    ]

    heat_slices = [
        heatmap[mid_heat[0], :, :],
        heatmap[:, mid_heat[1], :],
        heatmap[:, :, mid_heat[2]],
    ]

    overlays = []

    for img_slice, hm_slice in zip(slices, heat_slices):
        # Resize both to fixed output size
        img_r = cv2.resize(img_slice.astype(np.float32), (OUTPUT_SIZE, OUTPUT_SIZE))
        hm_r  = cv2.resize(hm_slice.astype(np.float32), (OUTPUT_SIZE, OUTPUT_SIZE))

        img_u8   = (img_r * 255).clip(0, 255).astype(np.uint8)
        hm_u8    = (hm_r  * 255).clip(0, 255).astype(np.uint8)

        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
        img_bgr  = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

        overlay  = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)
        overlays.append(overlay)

    return overlays[0], overlays[1], overlays[2]