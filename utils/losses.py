import tensorflow as tf

# ==========================================================
# Dice Coefficient
# ==========================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    return (
        2.0 * intersection + smooth
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + smooth
    )


# ==========================================================
# Dice Loss
# ==========================================================

def dice_loss(y_true, y_pred):

    return 1.0 - dice_coefficient(y_true, y_pred)


# ==========================================================
# Binary Cross Entropy Loss
# ==========================================================

bce = tf.keras.losses.BinaryCrossentropy()


# ==========================================================
# BCE + Dice Loss
# ==========================================================

def bce_dice_loss(y_true, y_pred):

    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)


# ==========================================================
# Binary Focal Loss
# ==========================================================

def binary_focal_loss(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
):

    y_true = tf.cast(y_true, tf.float32)

    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    bce_loss = -(
        y_true * tf.math.log(y_pred)
        + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    )

    p_t = (
        y_true * y_pred
        + (1.0 - y_true) * (1.0 - y_pred)
    )

    alpha_factor = (
        y_true * alpha
        + (1.0 - y_true) * (1.0 - alpha)
    )

    focal_weight = alpha_factor * tf.pow(
        1.0 - p_t,
        gamma,
    )

    return tf.reduce_mean(focal_weight * bce_loss)


# ==========================================================
# Focal + Dice Loss
# ==========================================================

def focal_dice_loss(y_true, y_pred):

    return (
        binary_focal_loss(y_true, y_pred)
        + dice_loss(y_true, y_pred)
    )


# ==========================================================
# Tversky Loss
# ==========================================================

def tversky_loss(
    y_true,
    y_pred,
    alpha=0.7,
    beta=0.3,
    smooth=1e-6,
):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    tp = tf.reduce_sum(y_true * y_pred)

    fp = tf.reduce_sum((1.0 - y_true) * y_pred)

    fn = tf.reduce_sum(y_true * (1.0 - y_pred))

    tversky = (
        tp + smooth
    ) / (
        tp
        + alpha * fp
        + beta * fn
        + smooth
    )

    return 1.0 - tversky


# ==========================================================
# Focal Tversky Loss
# ==========================================================

def focal_tversky_loss(
    y_true,
    y_pred,
    alpha=0.7,
    beta=0.3,
    gamma=0.75,
):

    loss = tversky_loss(
        y_true,
        y_pred,
        alpha,
        beta,
    )

    return tf.pow(loss, gamma)
# ==========================================================
# IoU Score
# ==========================================================

def iou_score(
    y_true,
    y_pred,
    smooth=1e-6,
):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    union = (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        - intersection
    )

    return (
        intersection + smooth
    ) / (
        union + smooth
    )
# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    y_true = tf.constant(
        [[1, 1, 0, 0]],
        dtype=tf.float32,
    )

    y_pred = tf.constant(
        [[0.9, 0.8, 0.2, 0.1]],
        dtype=tf.float32,
    )

    print("Dice            :", dice_coefficient(y_true, y_pred).numpy())
    print("Dice Loss       :", dice_loss(y_true, y_pred).numpy())
    print("BCE + Dice      :", bce_dice_loss(y_true, y_pred).numpy())
    print("Focal Loss      :", binary_focal_loss(y_true, y_pred).numpy())
    print("Focal + Dice    :", focal_dice_loss(y_true, y_pred).numpy())
    print("Tversky Loss    :", tversky_loss(y_true, y_pred).numpy())
    print("Focal Tversky   :", focal_tversky_loss(y_true, y_pred).numpy())
    print("IoU             :", iou_score(y_true, y_pred).numpy())