"""
Dataset Module
Custom dataset for loading CICIDS2017 CSV files.
Adapted from original VAEGAN CustomDataset.py
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils.data_transform import BinaryDataTransformer


class NetworkTrafficDataset(Dataset):
    """
    Custom Dataset for CICIDS2017 network traffic data.
    """
    
    def __init__(self, csv_files, feature_list=None, backup_data=None):
        """
        Args:
            csv_files: List of CSV file paths
            feature_list: List of feature column names to use
            backup_Fallback CSV file path for loading when files missing
        """
        self.csv_files = csv_files
        self.backup_file = backup_data
        self.feature_list = feature_list
        
        # Feature mapping (adjust for CSV columns)
        self.feature_map = {
            'Flow_Duration': 'Flow_Duration',
            'Total_Fwd_Packets': 'Total_Fwd_Packets',
            'Total_Bwd_Packets': 'Total_Bwd_Packets',
            'Flow_IAT_Min': 'Flow_IAT_Min',
            'Flow_IAT_Max': 'Flow_IAT_Max',
            'Flow_IAT_Mean': 'Flow_IAT_Mean',
            'Flow_IAT_Std': 'Flow_IAT_Std',
        }
        
        # Load data
        self.data_list = self._load_data()
        
        if len(self.data_list) > 0:
            self.num_variables = len(self.feature_list)
        else:
            self.num_variables = 0
    
    def _load_data(self):
        """Load data from all CSV files."""
        data_list = []
        
        for file_path in self.csv_files:
            if 'Padding' in file_path or 'padding' in file_path:
                continue
            
            try:
                df = pd.read_csv(file_path)
                data_list.append(df)
            except Exception as e:
                print(f"Warning loading {file_path}: {e}")
                continue
        
        # If no data found and backup exists, load backup
        if not data_list and self.backup_file:
            try:
                df = pd.read_csv(self.backup_file)
                data_list.append(df)
                print(f"Loaded backup file {self.backup_file}")
            except:
                pass
        
        return data_list
    
    def __len__(self):
        num_samples = sum(len(df) for df in self.data_list)
        return max(1, num_samples)
    
    def __getitem__(self, idx):
        """Get a sample - returns all data concatenated."""
        if not self.data_list:
            return None
        
        # Concatenate all loads
        all_dfs = self.data_list
        all_labels = []
        
        full_df = pd.DataFrame()
        for df in all_dfs:
            if 'Label' in df.columns:
                labels = df['Label'].values
                all_labels.append(labels)
        
        if all_labels:
            full_labels = np.concatenate(all_labels, axis=0)
        else:
            full_labels = None
        
        # Transform to binary representation
        # For now, return raw float data (transformation can be applied later)
        if 'Label' not in full_df.columns:
            return full_df.values, full_labels
        return full_df, full_labels


def create_data_transformer(feature_list):
    """Create a BinaryDataTransformer for the given features."""
    return BinaryDataTransformer(bit_length=32, frac_bits=12)


# Standard feature list
ALL_FEATURES = [
    'Flow_Duration', 'Total_Fwd_Packets', 'Total_Bwd_Packets',
    'Flow_IAT_Min', 'Flow_IAT_Max', 'Flow_IAT_Mean', 'Flow_IAT_Std',
    'Fwd_IAT_Min',
    'Fwd_IAT_Max', 'Fwd_IAT_Mean',
    'Fwd_IAT_Std',
    'Bwd_IAT_Min',
    'Bwd_IAT_Max', 'Bwd_IAT_Mean',
    'Bwd_IAT_Std',
    'Packets_Fwd',
    'Packets_Bwd',
    'Bytes_Fwd',
    'Bytes_Bwd',
    'Packet_Length_Mean',
    'Packet_Length_Variance',
    'Fwd_Packet_Length_Mean',
    'Bwd_Packet_Length_Mean',
    'Fwd_Packet_Length_Max',
    'Fwd_Packet_Length_Min',
    'Bwd_Packet_Length_Max',
    'Bwd_Packet_Length_Min',
    'Flow_Packets_Syn', 'Flow_Packets_Ack', 'Flow_Packets_URG',
    'Flow_Packets_FIN', 'Flow_Packets_RST', 'Flow_Packets_PSH',
    'FIN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count',
    'ACK_Flag_Count', 'URG_Flag_Count', ' CWE_Flag_Count',
    'ECE_Flag_Count', 'Label'
]