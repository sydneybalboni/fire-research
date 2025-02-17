import tensorflow as tf
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from wildfire_preprocess import create_sequences

# ============= CONFIGURATION =============
# Data parameters
FIRE_NAME = '2018/fire_21889697'
DATA_DIR = f"/data/ai_club/fire/WildfireSpreadLS/{FIRE_NAME}"
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
def build_unet(input_shape):
    # Your existing build_unet function here
    # ... (copy the entire function from your notebook)

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
        f.write("Model Evaluation Metrics:\n")
        f.write(f"{'Metric':<20} {'Value':<10}\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Loss':<20} {val_loss:.4f}\n")
        f.write(f"{'Accuracy':<20} {val_acc:.4f}\n")
        f.write(f"{'IoU Score':<20} {val_iou:.4f}\n")
        f.write(f"{'Precision':<20} {precision:.4f}\n")
        f.write(f"{'Recall':<20} {recall:.4f}\n")
        f.write(f"{'F1 Score':<20} {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{'':>10}{'Predicted':>20}\n")
        f.write(f"{'Actual':>10}{'No Fire':>10}{'Fire':>10}\n")
        f.write(f"{'No Fire':>10}{tn:>10.0f}{fp:>10.0f}\n")
        f.write(f"{'Fire':>10}{fn:>10.0f}{tp:>10.0f}\n")
    
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
    # Load and preprocess data
    file_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".tif")])
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
    class_weight = {0: 1.0, 1: pos_weight}
    
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
    
    # Train model
    history = model.fit(
        train_sequences,
        train_labels,
        validation_data=(val_sequences, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=True
    )
    
    # Plot and save results
    plot_training_history(history)
    predictions = evaluate_model(model, val_sequences, val_labels)
    plot_example_predictions(val_sequences, val_labels, predictions)

if __name__ == "__main__":
    main() 