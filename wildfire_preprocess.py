import numpy as np
import rasterio
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Fixed size for all images
FIXED_SIZE = (300, 220)

# ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

def load_tif(file_path):
    """ Load and normalize a single TIFF file with NaN handling """
    with rasterio.open(file_path) as src:
        bands = src.read()  # Shape: (23, H, W)
        normalized_bands = []
        for i in range(bands.shape[0]):
            band = np.nan_to_num(bands[i], nan=-1)  # Replace NaN with -1
            valid_mask = band != -1
            if valid_mask.sum() > 0:  # Normalize only if there are valid values
                band_min = band[valid_mask].min()
                band_max = band[valid_mask].max()
                band[valid_mask] = (band[valid_mask] - band_min) / (band_max - band_min + 1e-5)
            normalized_bands.append(band)
        bands = np.stack(normalized_bands, axis=0)
        bands = resize(bands, (23, *FIXED_SIZE), mode='reflect', preserve_range=True)
    return np.transpose(bands, (1, 2, 0))  # Shape: (H, W, 23)

def create_sequences(file_paths, sequence_length=3):
    """ Create input-output sequences from TIFF files with buffer day handling and temporal smoothing """
    sequences = []
    labels = []
    for i in range(len(file_paths) - sequence_length):
        input_sequence = [load_tif(file_paths[j]) for j in range(i, i + sequence_length)]
        label = load_tif(file_paths[i + sequence_length])[:, :, -1]  # Active fire band for prediction
        label = np.nan_to_num(label, nan=0)  # Replace NaN with 0 in labels
        label = temporal_smooth(label)
        if not is_buffer_day(i, len(file_paths), sequence_length):
            sequences.append(np.stack(input_sequence, axis=0))  # Shape: (sequence_length, H, W, 23)
            labels.append(label)
    return np.array(sequences), np.array(labels)

def temporal_smooth(label, window_size=3):
    """ Apply temporal smoothing to reduce noise in label sequences """
    smoothed = np.convolve(label.flatten(), np.ones(window_size) / window_size, mode='same')
    return smoothed.reshape(label.shape)

def is_buffer_day(day_index, total_days, sequence_length, buffer_days=4):
    """ Determine if the current day is within the buffer period """
    return day_index < buffer_days or day_index > (total_days - sequence_length - buffer_days)

# Example usage
if __name__ == "__main__":
    file_paths = ["/path/to/tif1.tif", "/path/to/tif2.tif", "/path/to/tif3.tif"]  # Replace with actual paths
    train_sequences, train_labels = create_sequences(file_paths)
    print("Train Sequences Shape:", train_sequences.shape)
    print("Train Labels Shape:", train_labels.shape)
