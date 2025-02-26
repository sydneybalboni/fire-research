import tensorflow as tf
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from wildfire_preprocess import create_sequences
from tqdm import tqdm

# ============= CONFIGURATION =============
# Data parameters
FIRE_NAME = '2018/fire_21889697'
DATASET_NAME = 'WildfireSpreadLS' # WildfireSpreadLS or 2018/fire_21889697
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
RESULTS_DIR = f"results/run_{TIMESTAMP}_{EXPERIMENT_NAME}"
LOSS_PLOT_NAME = "loss_history.png"
IOU_PLOT_NAME = "iou_history.png"
PREDICTIONS_PLOT_NAME = "example_predictions.png"
MODEL_SAVE_NAME = "best_model.h5"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============= MODEL DEFINITION =============
# Move loss functions to module level (before any function definitions)
def combined_loss(y_true, y_pred):
    # Focal loss component
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    focal = -FOCAL_ALPHA * y_true * tf.pow(1 - y_pred, FOCAL_GAMMA) * tf.math.log(y_pred) - \
            (1 - FOCAL_ALPHA) * (1 - y_true) * tf.pow(y_pred, FOCAL_GAMMA) * tf.math.log(1 - y_pred)
    
    # IoU loss component
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + epsilon) / (union + epsilon)
    iou_loss = 1 - iou
    
    # Combine losses
    return tf.reduce_mean(focal) + iou_loss

def build_unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Split inputs by feature groups for specialized processing
    # Indices based on the dataset description
    viirs_features = inputs[..., :3]  # VIIRS bands (I1, I2, M11)
    weather_features = inputs[..., 3:13]  # GRIDMET features
    forecast_features = inputs[..., 13:18]  # GFS features
    static_features = inputs[..., 18:]  # Land cover, elevation, etc.
    
    # Encoder with separate paths for different feature types
    def encoder_block(x, filters, name):
        x = tf.keras.layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same', name=f'{name}_conv1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = tf.keras.layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same', name=f'{name}_conv2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)
        skip = x
        x = tf.keras.layers.MaxPooling3D((1, 2, 2), name=f'{name}_pool')(x)
        return x, skip
    
    # Process different feature groups
    viirs_enc1, viirs_skip1 = encoder_block(viirs_features, 32, 'viirs1')
    viirs_enc2, viirs_skip2 = encoder_block(viirs_enc1, 64, 'viirs2')
    
    weather_enc1, weather_skip1 = encoder_block(weather_features, 32, 'weather1')
    weather_enc2, weather_skip2 = encoder_block(weather_enc1, 64, 'weather2')
    
    # Combine features
    combined = tf.keras.layers.Concatenate()([viirs_enc2, weather_enc2])
    
    # Bridge
    bridge = tf.keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(combined)
    bridge = tf.keras.layers.BatchNormalization()(bridge)
    bridge = tf.keras.layers.SpatialDropout3D(0.3)(bridge)  # Add dropout to prevent overfitting
    
    # Decoder with attention
    def attention_block(x, skip_connection):
        g1 = tf.keras.layers.Conv3D(x.shape[-1], (1, 1, 1))(skip_connection)
        g1 = tf.keras.layers.BatchNormalization()(g1)
        x1 = tf.keras.layers.Conv3D(x.shape[-1], (1, 1, 1))(x)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        psi = tf.keras.layers.Activation('relu')(g1 + x1)
        psi = tf.keras.layers.Conv3D(1, (1, 1, 1))(psi)
        psi = tf.keras.layers.Activation('sigmoid')(psi)
        return skip_connection * psi
    
    # Decoder
    def decoder_block(x, skip, filters):
        x = tf.keras.layers.Conv3DTranspose(filters, (3, 3, 3), strides=(1, 2, 2), padding='same')(x)
        attention = attention_block(x, skip)
        x = tf.keras.layers.Concatenate()([x, attention])
        x = tf.keras.layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x
    
    # Decoder path
    dec1 = decoder_block(bridge, tf.keras.layers.Concatenate()([viirs_skip2, weather_skip2]), 64)
    dec2 = decoder_block(dec1, tf.keras.layers.Concatenate()([viirs_skip1, weather_skip1]), 32)
    
    # Final convolution with class balancing
    outputs = tf.keras.layers.Conv3D(1, (3, 1, 1), padding='valid')(dec2)
    outputs = tf.keras.layers.Reshape((300, 220, 1))(outputs)
    outputs = tf.keras.layers.Activation('sigmoid')(outputs)
    
    model = tf.keras.Model(inputs, outputs)
    
    return model

def iou_metric(y_true, y_pred):
    # Your existing iou_metric function here
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())
    return iou

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
    
    # Calculate metrics
    val_loss, val_acc, val_iou = model.evaluate(val_sequences, val_labels, verbose=0)
    
    # Calculate additional metrics
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
        # Input sequence (last frame)
        axes[i, 0].imshow(val_sequences[i, -1, :, :, -1])
        axes[i, 0].set_title(f'Input (Last Frame)')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(val_labels[i])
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[i])
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, PREDICTIONS_PLOT_NAME))
    plt.close()

# ============= MAIN TRAINING SCRIPT =============
def main():
    # Set base directory to the 2018 directory
    base_dir = "/data/ai_club/fire/WildfireSpreadLS/2018"
    file_paths = []
    
    # Iterate over each fire directory within the 2018 directory
    for fire_dir in os.listdir(base_dir):
        fire_path = os.path.join(base_dir, fire_dir)
        
        # Collect all .tif files from the fire directory
        fire_files = [os.path.join(fire_path, f) for f in os.listdir(fire_path) if f.endswith(".tif")]
        file_paths.extend(fire_files)

    print(f"Total files found: {len(file_paths)}")
    
    # Create sequences
    train_sequences, train_labels = create_sequences(file_paths, sequence_length=SEQUENCE_LENGTH)
    
    # Split data
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_sequences, train_labels, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    # Add channel dimension to labels
    train_labels = np.expand_dims(train_labels, axis=-1)
    val_labels = np.expand_dims(val_labels, axis=-1)
    
    # Calculate class weights
    pos_weight = np.sum(train_labels == 0) / (np.sum(train_labels == 1) + 1e-6)
    
    # Build model
    model = build_unet(INPUT_SHAPE)
    
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
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: LEARNING_RATE * (0.95 ** epoch)
        )
    ]
    
    # Define a custom weighted loss function with closure over pos_weight
    def make_weighted_loss(pos_weight_value):
        def weighted_loss(y_true, y_pred):
            base_loss = combined_loss(y_true, y_pred)
            weights = tf.where(y_true > 0, tf.cast(pos_weight_value, tf.float32), 1.0)
            return base_loss * weights
        return weighted_loss
    
    # Compile model with weighted loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
        loss=make_weighted_loss(pos_weight),  # Create the loss function with pos_weight
        metrics=['accuracy', iou_metric]
    )
    
    # Train model
    history = model.fit(
        train_sequences,
        train_labels,
        validation_data=(val_sequences, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True
    )
    
    # Plot and save results
    plot_training_history(history)
    predictions = evaluate_model(model, val_sequences, val_labels)
    plot_example_predictions(val_sequences, val_labels, predictions)

if __name__ == "__main__":
    main() 