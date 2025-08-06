
import numpy as np
from typing import List, Union, Any, Optional, Dict, Tuple
from .models import DataType, CompressionType
from .compression import Compression



class Column:
    
    def __init__(self, 
                 name: str, 
                 dtype: DataType, 
                 data: Optional[Union[np.ndarray, List]] = None,
                 compression: CompressionType = CompressionType.NONE):
      
        self.name = name
        self.dtype = dtype
        self.numpy_dtype = Compression.get_numpy_dtype(dtype)
        self.compression = compression
        self.min_value = None
        self.max_value = None
        self.data: Optional[np.ndarray] = None
        self.compressed_data: Any = None
        
        if data is not None:
            self.set_data(data)
   
    def set_data(self, data: Union[np.ndarray, List]) -> None:
        """
        Set the column data and update statistics.
        
        Args:
            data: The data to set for the column.
        """
        if isinstance(data, list):
            self.data = np.array(data, dtype=self.numpy_dtype)
        else:
            # Ensure correct dtype if it's already a numpy array
            if data.dtype != self.numpy_dtype:
                self.data = data.astype(self.numpy_dtype)
            else:
                self.data = data
            
        # Update statistics for indexing
        if len(self.data) > 0:
            if self.dtype == DataType.STRING:
                self.min_value = min(self.data)
                self.max_value = max(self.data)
            elif self.dtype == DataType.DECIMAL:
                # For Decimal, ensure we're comparing the actual decimal values
                self.min_value = min(self.data)
                self.max_value = max(self.data)
            elif self.dtype == DataType.DATE or self.dtype == DataType.TIMESTAMP:
                # For datetime types, numpy min/max handle them correctly
                self.min_value = np.min(self.data)
                self.max_value = np.max(self.data)
                # Convert numpy datetime64 to integer for storage
                if self.dtype == DataType.TIMESTAMP:
                    self.min_value = int(self.min_value.astype('datetime64[ns]').astype(np.int64))
                    self.max_value = int(self.max_value.astype('datetime64[ns]').astype(np.int64))
                elif self.dtype == DataType.DATE:
                    self.min_value = int(self.min_value.astype('datetime64[D]').astype(np.int64))
                    self.max_value = int(self.max_value.astype('datetime64[D]').astype(np.int64))
            else:  # INT, FLOAT, BOOLEAN
                self.min_value = np.min(self.data)
                self.max_value = np.max(self.data)
        
        # Apply compression if specified
        if self.compression == CompressionType.AUTO:
            # Auto-select best compression using a comprehensive test
            self.compression = Compression.find_optimal_compression(self)
            
        if self.compression != CompressionType.NONE:
            self.compress()
        else:
            self.compressed_data = None
            
    def compress(self) -> None:
        """
        Apply compression to the column data.
        """
        Compression.compress(self)
        
    def decompress(self) -> Optional[np.ndarray]:
        """
        Decompress the column data.
        
        Returns:
            The decompressed data, or None if no compressed data is available.
        """
        return Compression.decompress(self)
        
    def compression_ratio(self) -> float:
        """
        Calculate the compression ratio for the column.
        
        Returns:
            The compression ratio (compressed size / original size). Lower is better.
        """
        return Compression.compression_ratio(self)

    def get_size(self) -> int:
        """
        Get the memory size of the column data.
        
        Returns:
            The memory size in bytes.
        """
        if self.data is None:
            return 0
        
        if self.compression != CompressionType.NONE and self.compressed_data is not None:
            # Calculate compressed size
            if self.compression == CompressionType.RLE:
                # For RLE, each entry is (value, count)
                if self.dtype == DataType.INT:
                    return sum(8 + 4 for _ in self.compressed_data)  # 8 bytes for value, 4 for count
                elif self.dtype == DataType.FLOAT:
                    return sum(8 + 4 for _ in self.compressed_data)  # 8 bytes for value, 4 for count
                else:  # STRING
                    return sum(len(str(value)) + 4 + 4 for value, _ in self.compressed_data)  # length of string + 4 for length + 4 for count
            elif self.compression == CompressionType.DICT:
                value_to_id, encoded_values = self.compressed_data
                dict_size = 0
                
                if isinstance(value_to_id, dict):
                    for value in value_to_id.keys():
                        if self.dtype == DataType.STRING:
                            dict_size += len(str(value)) + 4  # string length + 4 for id
                        else:
                            dict_size += 8 + 4  # 8 for value, 4 for id
                    
                    # Encoded values are just integers (4 bytes each)
                    return dict_size + len(encoded_values) * 4
        
        # Calculate uncompressed size
        if self.data is not None:
            if self.dtype == DataType.INT:
                return len(self.data) * 8  # 8 bytes per int64
            elif self.dtype == DataType.FLOAT:
                return len(self.data) * 8  # 8 bytes per float64
            else:  # STRING
                return sum(len(str(s)) + 4 for s in self.data)  # length of string + 4 for length
                
        return 0
    
