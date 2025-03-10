import numpy as np

def min_max_scaler(data, feature_range=(0, 1)):
    min_val, max_val = feature_range
    min_data = np.min(data)
    max_data = np.max(data)
    
    if max_data == min_data:
        return np.zeros_like(data) if min_val == 0 else np.full_like(data, min_val)
    
    scaled_data = (data - min_data) / (max_data - min_data) * (max_val - min_val) + min_val
    return scaled_data

data = np.array([10, 20, 30, 40, 50])
scaled_data = min_max_scaler(data, feature_range=(0, 1))
print("Original Data:", data)
print("Scaled Data:", scaled_data)