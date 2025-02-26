import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate,
    BatchNormalization, Activation, GlobalAveragePooling3D, Dense, Reshape, Multiply
)
import math

# --------------------------
# 🔥 1️⃣ Custom Loss Functions
# --------------------------
def focal_loss(alpha=0.25, gamma=2.0):
    """Focal Loss to handle class imbalance"""
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        focal_term = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return K.mean(focal_term * bce)
    return loss

def iou_loss(y_true, y_pred):
    """IoU loss to improve segmentation accuracy"""
    smooth = 1e-6
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3, 4])
    union = K.sum(y_true, axis=[1, 2, 3, 4]) + K.sum(y_pred, axis=[1, 2, 3, 4]) - intersection
    return 1 - K.mean((intersection + smooth) / (union + smooth))

# --------------------------
# 🔥 2️⃣ SE Block (Squeeze-and-Excitation)
# --------------------------
def se_block(input_tensor, reduction=16):
    """ Squeeze-and-Excitation block to enhance feature representation """
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling3D()(input_tensor)
    se = Dense(filters // reduction, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, 1, filters))(se)
    return Multiply()([input_tensor, se])

# --------------------------
# 🔥 3️⃣ 3D U-Net with Attention & Dilated Convolutions
# --------------------------
def conv_block(x, filters, kernel_size=3, dilation_rate=1, use_se=True):
    """ Convolutional Block with BatchNorm, ReLU, Dilated Conv, and optional SE Block """
    x = Conv3D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if use_se:
        x = se_block(x)
    return x

def build_3d_unet(input_shape=(64, 64, 64, 4)):
    """ 3D U-Net model with SE blocks and dilated convolutions """
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, 32, dilation_rate=1)
    p1 = MaxPooling3D(pool_size=(2, 2, 2))(c1)

    c2 = conv_block(p1, 64, dilation_rate=2)
    p2 = MaxPooling3D(pool_size=(2, 2, 2))(c2)

    c3 = conv_block(p2, 128, dilation_rate=4)
    p3 = MaxPooling3D(pool_size=(2, 2, 2))(c3)

    c4 = conv_block(p3, 256, dilation_rate=8)

    # Decoder
    u5 = UpSampling3D(size=(2, 2, 2))(c4)
    c5 = conv_block(Concatenate()([u5, c3]), 128, dilation_rate=4)

    u6 = UpSampling3D(size=(2, 2, 2))(c5)
    c6 = conv_block(Concatenate()([u6, c2]), 64, dilation_rate=2)

    u7 = UpSampling3D(size=(2, 2, 2))(c6)
    c7 = conv_block(Concatenate()([u7, c1]), 32, dilation_rate=1)

    outputs = Conv3D(1, kernel_size=1, activation='sigmoid')(c7)

    return Model(inputs, outputs)

# --------------------------
# 🔥 4️⃣ Learning Rate Scheduler (Cosine Annealing)
# --------------------------
def cosine_annealing(epoch, lr, T_max=50, eta_min=1e-6):
    """ Cosine Annealing LR Scheduler """
    return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch, lr: cosine_annealing(epoch, lr, 50))

# --------------------------
# 🔥 5️⃣ Enable Mixed Precision Training
# --------------------------
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# --------------------------
# 🔥 6️⃣ Load & Train Model
# --------------------------
# Placeholder dataset (Replace with actual dataset)
train_dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal([10, 64, 64, 64, 4]), tf.random.uniform([10, 64, 64, 64, 1], maxval=2, dtype=tf.float32)))
train_dataset = train_dataset.batch(2).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal([2, 64, 64, 64, 4]), tf.random.uniform([2, 64, 64, 64, 1], maxval=2, dtype=tf.float32)))
val_dataset = val_dataset.batch(2).repeat()

# Build & Compile Model
model = build_3d_unet()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=[focal_loss(), iou_loss],
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10, steps_per_epoch=5, validation_steps=2, callbacks=[lr_scheduler])

# Save model
model.save("wildfire_3d_unet.h5")

print("🔥 Model training complete and saved as 'wildfire_3d_unet.h5'!")