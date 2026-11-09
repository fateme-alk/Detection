"""
Data Transformation Module
Converts float/integer values to 32-bit fixed-point binary representation.
Adapted from original VAEGAN DataTransformation.py
"""

import numpy as np


class BinaryDataTransformer:
    """
    Binary Data Transformer class for network traffic features.
    
    This class handles:
    1. Normalization of input features
    2. Conversion to fixed-point binary representation
    3. Support for different feature scaling methods
    """
    
    def __init__(self, 
                 feature_stats=None, 
                 bit_length=32, 
                 frac_bits=12,
                 log_transform_features=None):
        """
        Args:
            feature_stats: Dictionary of min/max for each feature
            bit_length: Total bits for binary representation
            frac_bits: Bits for fractional part
            log_transform_features: List of features to apply log transform
        """
        self.bit_length = bit_length
        self.frac_bits = frac_bits
        self.feature_stats = feature_stats or {}
        self.log_transform_features = log_transform_features or []
        
        # Compute min/max from stats
        self.feature_mins = {}
        self.feature_maxs = {}
        
        for feat in self.feature_stats.keys():
            self.feature_mins[feat] = self.feature_stats[feat].get('min', -1.0)
            self.feature_maxs[feat] = self.feature_stats[feat].get('max', 1.0)
    
    def fit(self, data):
        """
        Compute statistics from training data.
        
        Args:
            DataFrame with features
        """
        # Log transform features first
        data_transformed = data.copy()
        for feat in self.log_transform_features:
            if feat in data_transformed.columns:
                # Add small epsilon to avoid log(0)
                data_transformed = data_transformed.copy()
                data_transformed[feat] = np.log1p(np.abs(data_transformed[feat]))
        
        # Compute min/max
        for feat in self.feature_stats.keys():
            if feat in data_transformed.columns:
                self.feature_mins[feat] = data_transformed[feat].min()
                self.feature_maxs[feat] = data_transformed[feat].max()
    
    def transform(self, data, include_label=True):
        """
        Transform data to binary representation.
        
        Args:
            DataFrame with features
            include_label: Whether to include Label column
            
        Returns:
            Binary data as numpy array (B, num_features, bit_length)
        """
        features = self.feature_stats.keys() if include_label else [f for f in self.feature_stats.keys() if f != 'Label']
        
        num_samples = len(data)
        num_features = len(features)
        
        # Initialize binary array
        binary_data = np.zeros((num_samples, num_features, self.bit_length), dtype=np.uint8)
        
        for idx, feat in enumerate(features):
            if feat not in data.columns:
                continue
            
            values = data[feat].values
            
            # Log transform
            if feat in self.log_transform_features:
                values = np.log1p(values)
            
            # Normalize to [0, 1]
            min_val = self.feature_mins.get(feat, 0.0)
            max_val = self.feature_maxs.get(feat, 1.0)
            
            # Handle edge case
            if max_val - min_val < 1e-10:
                normalized = np.zeros_like(values)
            else:
                normalized = (values - min_val) / (max_val - min_val)
            
            # Clip to [0, 1]
            normalized = np.clip(normalized, 0, 1)
            
            # Convert to binary fractional bits
            num_frac_bits = self.frac_bits
            scaled = (normalized * (2 ** num_frac_bits)).astype(np.uint8)
            
            # Fill integer bits with a deterministic pattern
            num_int_bits = self.bit_length - 1 - num_frac_bits
            binary_data[:, idx, :num_int_bits] = np.tile(
                np.arange(num_int_bits, dtype=np.uint8), 
                (num_samples, 1)
            ).astype(np.uint8)
            
            # Fill fractional bits
            binary_data[:, idx, num_int_bits:num_int_bits+num_frac_bits] = scaled
        
        return binary_data


def create_csv_from_binary(binary_data, feature_names, labels=None, output_path=None):
    """
    Create CSV files from binary data for output.
    
    Args:
        binary_Binary tensor (B, 1, num_features, bit_length)
        feature_names: List of feature names
        labels: Optional array of labels (if generated data)
        output_path: Output path prefix
    """
    import pandas as pd
    
    num_samples = binary_data.shape[0]
    num_vars = binary_data.shape[2]
    
    binary_ints = binary_data.reshape(num_samples, num_vars, -1)
    
    data_dict = {}
    
    for i, feat in enumerate(feature_names):
        col_name = f"{feat}_bin"
        data_dict[col_name] = binary_ints[:, i, :]
        
        # Create float representation
        binary_str = ''.join(str(int(x)) for x in binary_ints[0, i, :])
        val = sum(int(b) * (2 ** i) for i, b in enumerate(reversed(binary_str[:20])))
        col_name_float = f"{feat}_float"
        data_dict[col_name_float] = [val / (2 ** 20) for _ in range(num_samples)]
    
    if labels is not None:
        data_dict['Label'] = labels
    
    df = pd.DataFrame(data_dict)
    df.to_csv(output_path, index=False)
    print(f"Saved {num_samples} samples to {output_path}")