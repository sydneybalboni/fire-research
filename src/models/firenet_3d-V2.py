import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from preprocess.wildfire_preprocess import create_sequences
from tensorflow.keras.layers import GaussianNoise

# ============= CONFIGURATION =============
DEBUG=True # Enable if you want extra print statements for what it is doing in slurm.out file

# Data parameters
DATASET_NAME = 'WildfireSpreadLS'
YEAR = "2018" # Specific year OR None or "" for all years
SEQUENCE_LENGTH = 3
INPUT_SHAPE = (3, 300, 220, 23)

# Training hyperparameters
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
CLIPNORM = 1.0
VALIDATION_SPLIT = 0.2

# Loss function parameters
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# Early stopping parameters
PATIENCE = 10
MIN_DELTA = 1e-4

# Results configuration
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = (
    f"b{BATCH_SIZE}"           # batch size
    f"_e{EPOCHS}"             # epochs
    f"_lr{LEARNING_RATE:.0e}"  # learning rate in scientific notation
    f"_c{CLIPNORM:.1f}"       # clip norm
    f"_a{FOCAL_ALPHA:.2f}"    # focal alpha
    f"_g{FOCAL_GAMMA:.1f}"    # focal gamma
    f"_p{PATIENCE}"           # early stopping patience
    f"_v{VALIDATION_SPLIT:.1f}"  # validation split
)
RESULTS_DIR = f"../results/run_{TIMESTAMP}_{EXPERIMENT_NAME}"
LOSS_PLOT_NAME = "loss_history.png"
IOU_PLOT_NAME = "iou_history.png"
PREDICTIONS_PLOT_NAME = "example_predictions.png"
MODEL_SAVE_NAME = f"MODEL-{EXPERIMENT_NAME}.h5"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============= LOSS & METRIC DEFINITIONS =============

def combined_loss_with_posweight(y_true, y_pred, pos_weight=1.0):
    """
    Combined focal + IoU loss, with pixel-wise weighting for the positive class.
    """
    epsilon = tf.keras.backend.epsilon()
    
    # Clip predictions to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    # ---- Focal loss (per-pixel) ----
    # standard focal per pixel:
    focal_pos = -FOCAL_ALPHA * y_true * tf.pow(1 - y_pred, FOCAL_GAMMA) * tf.math.log(y_pred)
    focal_neg = -(1 - FOCAL_ALPHA) * (1 - y_true) * tf.pow(y_pred, FOCAL_GAMMA) * tf.math.log(1 - y_pred)
    focal_per_pixel = focal_pos + focal_neg

    # pixel-wise weighting for positives
    # if y_true=1, multiply by pos_weight, else multiply by 1.0
    weight_map = 1.0 + y_true * (pos_weight - 1.0)  # shape = same as y_true
    focal_per_pixel = focal_per_pixel * weight_map

    # average the focal loss across all pixels/batch
    focal_loss = tf.reduce_mean(focal_per_pixel)

    # ---- IoU loss (global) ----
    # We do the usual intersection-over-union across entire batch
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    iou_loss = 1 - iou

    return focal_loss + iou_loss


def iou_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())
    return iou


# ============= MODEL DEFINITION =============

def build_unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Data augmentation
    augmented = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(inputs)
    augmented = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(augmented)
    augmented = GaussianNoise(0.1)(augmented)
    
    # Split inputs by feature groups for specialized processing
    # Indices based on the dataset description
    viirs_features    = augmented[..., :3]    # VIIRS bands (I1, I2, M11)
    weather_features  = augmented[..., 3:13]  # GRIDMET features
    forecast_features = augmented[..., 13:18] # GFS features
    static_features   = augmented[..., 18:]   # Land cover, elevation, etc.
    
    def encoder_block(x, filters, name):
        x = tf.keras.layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_conv1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = tf.keras.layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=f'{name}_conv2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)
        skip = x
        x = tf.keras.layers.MaxPooling3D((1, 2, 2), name=f'{name}_pool')(x)
        return x, skip
    
    # ---------------- ENCODERS ----------------
    # VIIRS
    viirs_enc1, viirs_skip1 = encoder_block(viirs_features, 32, 'viirs1')
    viirs_enc2, viirs_skip2 = encoder_block(viirs_enc1, 64, 'viirs2')
    viirs_enc3, viirs_skip3 = encoder_block(viirs_enc2, 128, 'viirs3')
    
    # WEATHER
    weather_enc1, weather_skip1 = encoder_block(weather_features, 32, 'weather1')
    weather_enc2, weather_skip2 = encoder_block(weather_enc1, 64, 'weather2')
    weather_enc3, weather_skip3 = encoder_block(weather_enc2, 128, 'weather3')

    # FORECAST
    forecast_enc1, forecast_skip1 = encoder_block(forecast_features, 32, 'forecast1')
    forecast_enc2, forecast_skip2 = encoder_block(forecast_enc1, 64, 'forecast2')
    forecast_enc3, forecast_skip3 = encoder_block(forecast_enc2, 128, 'forecast3')

    # STATIC
    static_enc1, static_skip1 = encoder_block(static_features, 32, 'static1')
    static_enc2, static_skip2 = encoder_block(static_enc1, 64, 'static2')
    static_enc3, static_skip3 = encoder_block(static_enc2, 128, 'static3')
    
    # Combine features from all 4 branches
    combined = tf.keras.layers.Concatenate()([
        viirs_enc3, weather_enc3, forecast_enc3, static_enc3
    ])
    
    # ---------------- BRIDGE ----------------
    bridge = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(combined)
    bridge = tf.keras.layers.BatchNormalization()(bridge)
    bridge = tf.keras.layers.SpatialDropout3D(0.3)(bridge)  # Add dropout to prevent overfitting
    
    # ---------------- DECODER WITH ATTENTION ----------------
    def attention_block(x, skip_connection):
        # gating: match channels
        g1 = tf.keras.layers.Conv3D(x.shape[-1], (1, 1, 1))(skip_connection)
        g1 = tf.keras.layers.BatchNormalization()(g1)
        x1 = tf.keras.layers.Conv3D(x.shape[-1], (1, 1, 1))(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        psi = tf.keras.layers.Activation('relu')(g1 + x1)
        psi = tf.keras.layers.Conv3D(1, (1, 1, 1))(psi)
        psi = tf.keras.layers.Activation('sigmoid')(psi)
        return skip_connection * psi
    
    def decoder_block(x, skip, filters):
        x = tf.keras.layers.Conv3DTranspose(filters, (3, 3, 3), strides=(1, 2, 2), padding='same')(x)
        attention = attention_block(x, skip)
        x = tf.keras.layers.Concatenate()([x, attention])
        x = tf.keras.layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x
    
    # Decoder steps (matching how we combined in the encoder)
    dec1_input = tf.keras.layers.Concatenate()([
        viirs_skip3, weather_skip3, forecast_skip3, static_skip3
    ])
    dec1 = decoder_block(bridge, dec1_input, 128)

    dec2_input = tf.keras.layers.Concatenate()([
        viirs_skip2, weather_skip2, forecast_skip2, static_skip2
    ])
    dec2 = decoder_block(dec1, dec2_input, 64)

    dec3_input = tf.keras.layers.Concatenate()([
        viirs_skip1, weather_skip1, forecast_skip1, static_skip1
    ])
    dec3 = decoder_block(dec2, dec3_input, 32)

    # ---------------- OUTPUT LAYER ----------------
    # Use a 3D conv that reduces time dimension from 3->1 if 'valid' is used 
    # (since kernel_size=(3,1,1)), or keep 'same' if you want the same # of frames out.
    outputs = tf.keras.layers.Conv3D(1, (3, 1, 1), padding='valid')(dec3)
    # flatten out the time dimension, shape: (batch, 1, 300, 220, 1) -> (batch, 300, 220, 1)
    outputs = tf.keras.layers.Reshape((300, 220, 1))(outputs)
    
    outputs = tf.keras.layers.Activation('sigmoid')(outputs)
    
    model = tf.keras.Model(inputs, outputs)
    return model


# ============= VISUALIZATION FUNCTIONS =============

def plot_training_history(history):
    """Plot and save training metrics"""
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, LOSS_PLOT_NAME))
    plt.close()
    
    # IoU plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['iou_metric'], label='Training IoU')
    plt.plot(history.history['val_iou_metric'], label='Validation IoU')
    plt.title('Model IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, IOU_PLOT_NAME))
    plt.close()


def evaluate_model(model, val_sequences, val_labels):
    """Evaluate model and save metrics"""
    predictions = model.predict(val_sequences)
    predictions_binary = (predictions > 0.5).astype(np.float32)
    
    # Evaluate on the validation set
    val_loss, val_acc, val_iou = model.evaluate(val_sequences, val_labels, verbose=0)
    
    # Additional metrics
    tp = np.sum(predictions_binary * val_labels)
    fp = np.sum(predictions_binary * (1 - val_labels))
    fn = np.sum((1 - predictions_binary) * val_labels)
    tn = np.sum((1 - predictions_binary) * (1 - val_labels))
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    # Save metrics to file
    metrics_file = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Model Evaluation Metrics:\n")
        f.write(f"{'Metric':<20} {'Value':<10}\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Loss':<20} {val_loss:.4f}\n")
        f.write(f"{'Accuracy':<20} {val_acc:.4f}\n")
        f.write(f"{'IoU Score':<20} {val_iou:.4f}\n")
        f.write(f"{'Precision':<20} {precision:.4f}\n")
        f.write(f"{'Recall':<20} {recall:.4f}\n")
        f.write(f"{'F1 Score':<20} {f1:.4f}\n\n")
        
    return predictions_binary


def plot_example_predictions(val_sequences, val_labels, predictions, num_examples=3):
    """Plot and save example predictions"""
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))
    
    for i in range(num_examples):
        # Input sequence (last frame) - e.g., show the last channel or last time-slice
        # This is just an example of how you might visualize the data
        axes[i, 0].imshow(val_sequences[i, -1, :, :, 0], cmap='gray')
        axes[i, 0].set_title(f'Input (Last Frame, Channel=0)')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(val_labels[i].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction (>0.5)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, PREDICTIONS_PLOT_NAME))
    plt.close()


# ============= MAIN TRAINING SCRIPT =============
def main():
    if DEBUG:
        print("STARTING IN DEBUG MODE")

    base_dir = f"/data/ai_club/fire/{DATASET_NAME}"

    file_paths = []

    if DEBUG:
        print("FINDING YEARS...")

    # Determine which directories to process
    if YEAR:
        years_to_process = [YEAR]  # Only process the specified year
    else:
        years_to_process = [d for d in os.listdir(base_dir) if d.isdigit()]  # Get all year folders

    if DEBUG:
        print("YEARS FOUND: ", years_to_process)

    # Iterate over each year and fire directory
    for year_dir in years_to_process:
        if DEBUG:
            print("FINDING FIRES IN: ", year_dir)

        year_path = os.path.join(base_dir, year_dir)

        # Iterate over each fire directory within the year directory
        for fire_dir in os.listdir(year_path):
            fire_path = os.path.join(year_path, fire_dir)

            # Collect all .tif files from the fire directory
            fire_files = [os.path.join(fire_path, f) for f in os.listdir(fire_path) if f.endswith(".tif")]
            file_paths.extend(fire_files)

        if DEBUG:
            print("FINISHED FINDING FIRES IN: ", year_dir)

    print(f"Total files found: {len(file_paths)}")

    if DEBUG:
        print("FINISHED FINDING YEARS")
        print("CREATING SEQUENCES...")

    # Create sequences
    train_sequences, train_labels = create_sequences(file_paths, sequence_length=SEQUENCE_LENGTH)

    if DEBUG:
        print("FINISHED CREATING SEQUENCES")
        print("SPLITTING DATA...")

    # Split data
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_sequences, train_labels, test_size=VALIDATION_SPLIT, random_state=42
    )

    if DEBUG:
        print("FINISHED SPLITTING DATA")
    
    # Add channel dimension to labels
    train_labels = np.expand_dims(train_labels, axis=-1)
    val_labels   = np.expand_dims(val_labels, axis=-1)
    
    # Calculate class weights
    # pos_weight is typically (# of negative samples) / (# of positive samples).
    # We'll pass it into combined_loss_with_posweight.
    num_pos = np.sum(train_labels == 1)
    num_neg = np.sum(train_labels == 0)
    pos_weight = (num_neg / (num_pos + 1e-7))
    print(f"Positive samples: {num_pos}, Negative samples: {num_neg}, pos_weight: {pos_weight:.2f}")

    if DEBUG:
        print("BUILDING MODEL...")

    # Build model
    model = build_unet(INPUT_SHAPE)

    if DEBUG:
        print("FINISHED BUILDING MODEL")
        print("DEFINING CALLBACKS...")
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_iou_metric',
            patience=PATIENCE,
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_iou_metric',
            factor=0.5,
            patience=5,
            mode='max',
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, MODEL_SAVE_NAME),
            monitor='val_iou_metric',
            mode='max',
            save_best_only=True
        ),
        # REMOVED the LearningRateScheduler to avoid conflict with ReduceLROnPlateau
    ]

    if DEBUG:
        print("FINISHED DEFINING CALLBACKS")
        print("COMPILING MODEL...")
    
    # Compile model with your combined loss (with pos weight)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
        loss=lambda y_true, y_pred: combined_loss_with_posweight(y_true, y_pred, pos_weight=pos_weight),
        metrics=['accuracy', iou_metric]
    )

    if DEBUG:
        print("FINISHED COMPILING MODEL")
    
    # Train model
    print(f"Starting training, will save to {RESULTS_DIR}")
    history = model.fit(
        train_sequences,
        train_labels,
        validation_data=(val_sequences, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True
    )
    
    print("Training complete, saving results...")
    # Plot and save results
    plot_training_history(history)
    predictions = evaluate_model(model, val_sequences, val_labels)
    plot_example_predictions(val_sequences, val_labels, predictions)
    print(f"Results saved to {RESULTS_DIR}")

    
if __name__ == "__main__":
    main()
