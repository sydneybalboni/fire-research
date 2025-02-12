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
    """ Load and normalize a single TIFF file with specific handling for each data type """
    with rasterio.open(file_path) as src:
        bands = src.read()  # Shape: (23, H, W)
        normalized_bands = []
        
        # Get band descriptions for specific handling
        band_descriptions = src.descriptions
        
        for i in range(bands.shape[0]):
            band = bands[i]
            band_name = band_descriptions[i] if band_descriptions else f"Band_{i}"
            
            # Special handling for different data types
            if "active fire" in band_name.lower():
                # Binary classification for active fire (0 or 1)
                band = np.nan_to_num(band, nan=0)
                band = (band > 0).astype(np.float32)
            
            elif any(x in band_name.lower() for x in ["ndvi", "evi2"]):
                # Vegetation indices typically range from -1 to 1
                band = np.nan_to_num(band, nan=0)
                band = np.clip(band, -1, 1)
                band = (band + 1) / 2  # Normalize to [0,1]
            
            elif "land cover" in band_name.lower():
                # Land cover is categorical (1-17 in IGBP classification)
                band = np.nan_to_num(band, nan=0)
                band = band / 17.0  # Normalize by number of categories
            
            elif any(x in band_name.lower() for x in ["slope", "aspect", "elevation"]):
                # Topographical features
                band = np.nan_to_num(band, nan=band[~np.isnan(band)].mean())
                if "aspect" in band_name.lower():
                    band = band / 360.0  # Normalize degrees to [0,1]
                else:
                    band = (band - band.min()) / (band.max() - band.min() + 1e-8)
            
            elif "pdsi" in band_name.lower():
                # Palmer Drought Severity Index typically ranges from -10 to 10
                band = np.nan_to_num(band, nan=0)
                band = np.clip(band, -10, 10)
                band = (band + 10) / 20  # Normalize to [0,1]
            
            else:
                # Default normalization for other bands
                band = np.nan_to_num(band, nan=band[~np.isnan(band)].mean())
                if band.max() != band.min():
                    band = (band - band.min()) / (band.max() - band.min() + 1e-8)
                
            normalized_bands.append(band)
            
        bands = np.stack(normalized_bands, axis=0)
        bands = resize(bands, (23, *FIXED_SIZE), mode='reflect', preserve_range=True)
    return np.transpose(bands, (1, 2, 0))  # Shape: (H, W, 23)

def create_sequences(file_paths, sequence_length=3):
    """ Create input-output sequences with proper handling of temporal aspects """
    sequences = []
    labels = []
    
    for i in range(len(file_paths) - sequence_length):
        # Skip if we're in the buffer period (4 days before/after fire event)
        if is_buffer_day(i, len(file_paths), sequence_length):
            continue
            
        # Load sequence of input days
        input_sequence = [load_tif(file_paths[j]) for j in range(i, i + sequence_length)]
        
        # Get active fire band for next day prediction
        label = load_tif(file_paths[i + sequence_length])[:, :, -1]  # Active fire is last band
        
        # Ensure binary values for active fire prediction
        label = (label > 0).astype(np.float32)
        
        sequences.append(np.stack(input_sequence, axis=0))
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

def temporal_smooth(label, window_size=3):
    """ Apply temporal smoothing to reduce noise in label sequences """
    smoothed = np.convolve(label.flatten(), np.ones(window_size) / window_size, mode='same')
    return smoothed.reshape(label.shape)

def is_buffer_day(day_index, total_days, sequence_length, buffer_days=4):
    """ Handle buffer periods before and after fire events """
    return day_index < buffer_days or day_index > (total_days - sequence_length - buffer_days)

# Example usage
if __name__ == "__main__":
    file_paths = ["/path/to/tif1.tif", "/path/to/tif2.tif", "/path/to/tif3.tif"]  # Replace with actual paths
    train_sequences, train_labels = create_sequences(file_paths)
    print("Train Sequences Shape:", train_sequences.shape)
    print("Train Labels Shape:", train_labels.shape)
