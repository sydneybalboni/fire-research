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
        band_descriptions = src.descriptions
        
        for i in range(bands.shape[0]):
            band = bands[i]
            band_name = band_descriptions[i] if band_descriptions else f"Band_{i}"
            
            # Check if band is entirely NaN
            if np.all(np.isnan(band)):
                print(f"Warning: Band {band_name} is entirely NaN")
                band = np.zeros_like(band)
            
            # VIIRS Surface Reflectance (I1, I2, M11)
            if any(x in band_name.upper() for x in ['I1', 'I2', 'M11']):
                if np.any(np.isnan(band)):
                    median = np.nanmedian(band)
                    # If median is NaN (all values are NaN), set to 0
                    median = 0 if np.isnan(median) else median
                    band = np.nan_to_num(band, nan=median)
                
                # Safe min-max normalization
                min_val = np.nanmin(band)
                max_val = np.nanmax(band)
                if min_val == max_val:
                    band = np.zeros_like(band)
                else:
                    band = (band - min_val) / (max_val - min_val + 1e-8)
            
            # Vegetation indices (NDVI, EVI2)
            elif any(x in band_name.upper() for x in ['NDVI', 'EVI2']):
                band = np.nan_to_num(band, nan=0)
                band = np.clip(band, -1, 1)
                band = (band + 1) / 2
            
            # GRIDMET weather data
            elif any(x in band_name.lower() for x in ['temperature', 'precipitation', 'wind', 'humidity']):
                if np.any(np.isnan(band)):
                    median = np.nanmedian(band)
                    median = 0 if np.isnan(median) else median
                    band = np.nan_to_num(band, nan=median)
                
                # Safe min-max normalization
                min_val = np.nanmin(band)
                max_val = np.nanmax(band)
                if min_val == max_val:
                    band = np.zeros_like(band)
                else:
                    band = (band - min_val) / (max_val - min_val + 1e-8)
            
            # Palmer Drought Severity Index (PDSI)
            elif 'pdsi' in band_name.lower():
                band = np.nan_to_num(band, nan=0)
                band = np.clip(band, -10, 10)
                band = (band + 10) / 20
            
            # Land cover (IGBP classification)
            elif 'LC_Type1' in band_name:
                band = np.nan_to_num(band, nan=0)
                band = np.clip(band, 0, 17)  # Ensure valid class range
                band = band / 17.0
            
            # Topography data (elevation, slope, aspect)
            elif any(x in band_name.lower() for x in ['elevation', 'slope', 'aspect']):
                if np.any(np.isnan(band)):
                    median = np.nanmedian(band)
                    median = 0 if np.isnan(median) else median
                    band = np.nan_to_num(band, nan=median)
                
                if 'aspect' in band_name.lower():
                    band = np.clip(band, 0, 360)
                    band = band / 360.0
                else:
                    # Safe min-max normalization
                    min_val = np.nanmin(band)
                    max_val = np.nanmax(band)
                    if min_val == max_val:
                        band = np.zeros_like(band)
                    else:
                        band = (band - min_val) / (max_val - min_val + 1e-8)
            
            # Active fire (binary classification)
            elif 'active fire' in band_name.lower():
                band = np.nan_to_num(band, nan=0)
                band = (band > 0).astype(np.float32)
            
            # Default normalization for other bands
            else:
                if np.any(np.isnan(band)):
                    median = np.nanmedian(band)
                    median = 0 if np.isnan(median) else median
                    band = np.nan_to_num(band, nan=median)
                
                # Safe min-max normalization
                min_val = np.nanmin(band)
                max_val = np.nanmax(band)
                if min_val == max_val:
                    band = np.zeros_like(band)
                else:
                    band = (band - min_val) / (max_val - min_val + 1e-8)
            
            normalized_bands.append(band)
        
        bands = np.stack(normalized_bands, axis=0)
        bands = resize(bands, (23, *FIXED_SIZE), mode='reflect', preserve_range=True)
    return np.transpose(bands, (1, 2, 0))

def create_sequences(file_paths, sequence_length=3):
    """ Create input-output sequences with proper handling of temporal aspects """
    sequences = []
    labels = []
    
    # Sort file paths by date to ensure temporal order
    file_paths = sorted(file_paths)
    
    for i in range(len(file_paths) - sequence_length):
        # Skip if we don't have enough days before/after (4 days buffer as mentioned in description)
        if i < 4 or i > (len(file_paths) - sequence_length - 4):
            continue
        
        # Load sequence of input days
        input_sequence = []
        for j in range(i, i + sequence_length):
            day_data = load_tif(file_paths[j])
            input_sequence.append(day_data)
        
        # Get active fire band for next day prediction
        label = load_tif(file_paths[i + sequence_length])[:, :, -1]
        label = (label > 0).astype(np.float32)  # Ensure binary values
        
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
