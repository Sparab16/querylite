from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from .models import DataType, CompressionType

class Compression:
    
    @classmethod
    def get_numpy_dtype(cls, data_type: DataType) -> np.dtype:
        """
        Convert a DataType enum value to a numpy dtype.
        
        Args:
            data_type: The DataType enum value.
            
        Returns:
            The corresponding numpy dtype.
        """
        if data_type == DataType.INT:
            return np.dtype(np.int64)
        elif data_type == DataType.FLOAT:
            return np.dtype(np.float64)
        elif data_type == DataType.STRING:
            return np.dtype('O')  # Object type for strings
        elif data_type == DataType.BOOLEAN:
            return np.dtype(np.bool_)
        elif data_type == DataType.TIMESTAMP:
            return np.dtype('datetime64[ns]')
        elif data_type == DataType.DATE:
            return np.dtype('datetime64[D]')
        elif data_type == DataType.DECIMAL:
            return np.dtype('O')  # Object type for Decimal
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    
    @classmethod
    def compression_ratio(cls, column) -> float:
        """
        Calculate the compression ratio for a column.
        
        Args:
            column: The column to calculate compression ratio for.
            
        Returns:
            The compression ratio (compressed_size / original_size). Lower is better.
        """
        if column.data is None or column.compressed_data is None:
            return 1.0
            
        # Calculate original data size
        if column.dtype == DataType.STRING:
            # For strings, calculate size including string lengths
            original_size = 0
            for s in column.data:
                original_size += len(str(s)) + 8  # Pointer overhead
        else:
            # For numeric types, use array itemsize
            original_size = column.data.nbytes
        
        # Calculate compressed size
        compressed_size = 0
        
        if column.compression == CompressionType.RLE:
            values = column.compressed_data.get("values", np.array([]))
            counts = column.compressed_data.get("counts", np.array([]))
            compressed_size = values.nbytes + counts.nbytes
            
        elif column.compression == CompressionType.DICT:
            values = column.compressed_data.get("values", np.array([]))
            indices = column.compressed_data.get("indices", np.array([]))
            no_compression = column.compressed_data.get("no_compression", False)
            
            if no_compression:
                # Just return 1.0 for no compression case
                return 1.0
                
            if column.dtype == DataType.STRING:
                val_size = 0
                for s in values:
                    val_size += len(str(s)) + 8  # String + pointer overhead
                
                # Indices are 32-bit integers
                indices_size = len(indices) * 4
                compressed_size = val_size + indices_size
            else:
                compressed_size = values.nbytes + indices.nbytes
                
        elif column.compression == CompressionType.DELTA:
            base = column.compressed_data.get("base")
            deltas = column.compressed_data.get("deltas", np.array([]))
            compressed_size = 8 + deltas.nbytes  # 8 bytes for base value
            
        elif column.compression == CompressionType.BIT_PACK:
            packed = column.compressed_data.get("packed", bytearray())
            compressed_size = len(packed) + 16  # 16 bytes for metadata
            
        elif column.compression == CompressionType.PREFIX:
            prefixes = column.compressed_data.get("prefixes", np.array([]))
            suffixes = column.compressed_data.get("suffixes", np.array([]))
            indices = column.compressed_data.get("sorted_indices", np.array([]))
            
            suffix_size = 0
            for s in suffixes:
                suffix_size += len(str(s)) + 8
                
            compressed_size = prefixes.nbytes + suffix_size + indices.nbytes
            
        elif column.compression == CompressionType.ZSTD or column.compression == CompressionType.LZ4:
            compressed = column.compressed_data.get("compressed", b'')
            compressed_size = len(compressed)
        
        # Return ratio (lower is better)
        return compressed_size / max(1, original_size)
        
    @classmethod
    def compress(cls, column) -> None:
        """
        Apply the specified compression algorithm to the column data.
        
        Args:
            column: The column to compress.
        """
        if column.data is None or len(column.data) == 0:
            column.compressed_data = None
            return
            
        if column.compression == CompressionType.RLE:
            column.compressed_data = cls._run_length_encode(column)
        elif column.compression == CompressionType.DICT:
            column.compressed_data = cls._dictionary_encode(column)
        elif column.compression == CompressionType.DELTA:
            column.compressed_data = cls._delta_encode(column)
        elif column.compression == CompressionType.BIT_PACK:
            column.compressed_data = cls._bit_pack_encode(column)
        elif column.compression == CompressionType.PREFIX:
            column.compressed_data = cls._prefix_encode(column)
        elif column.compression == CompressionType.ZSTD or column.compression == CompressionType.LZ4:
            # Fall back to zlib for ZSTD and LZ4 since they require external libraries
            column.compressed_data = cls._zlib_encode(column)
    
    @classmethod
    def decompress(cls, column) -> Optional[np.ndarray]:
        """
        Decompress the column data.
        
        Args:
            column: The column to decompress.
            
        Returns:
            The decompressed column data, or None if no data is available.
        """
        if column.compression == CompressionType.NONE or column.compressed_data is None:
            return column.data
        
        if column.compression == CompressionType.RLE:
            return cls._run_length_decode(column)
        elif column.compression == CompressionType.DICT:
            return cls._dictionary_decode(column)
        elif column.compression == CompressionType.DELTA:
            return cls._delta_decode(column)
        elif column.compression == CompressionType.BIT_PACK:
            return cls._bit_pack_decode(column)
        elif column.compression == CompressionType.PREFIX:
            return cls._prefix_decode(column)
        elif column.compression == CompressionType.ZSTD or column.compression == CompressionType.LZ4:
            return cls._zlib_decode(column)
        
        return None

    @classmethod
    def find_optimal_compression(cls, column) -> CompressionType:
        """
        Find the optimal compression algorithm by testing all applicable algorithms
        and choosing the one with the best compression ratio.
        
        Returns:
            The CompressionType that achieved the best compression.
        """
        if column.data is None or len(column.data) == 0:
            return CompressionType.NONE
            
        # For small datasets, compression might not be worth it
        if len(column.data) < 100:
            return CompressionType.NONE
            
        # For very large datasets, be more selective about compression options to test
        # to avoid excessive processing time
        is_large_dataset = len(column.data) > 100000
            
        # Store original state
        original_data = column.data
        best_ratio = 0.95  # Only use compression if we get at least 5% reduction
        best_compression = CompressionType.NONE
        
        # List of compression types to test
        compression_types = []
        
        # Select which compression types to test based on the data type
        if column.dtype == DataType.STRING:
            # For string columns, analyze uniqueness first
            unique_ratio = len(set(column.data[:min(10000, len(column.data))])) / min(10000, len(column.data))
            
            if unique_ratio > 0.8 and is_large_dataset:
                # For strings with high uniqueness, compression likely won't help much
                return CompressionType.NONE
                
            compression_types = [CompressionType.DICT]
            
            # Only add these if the dataset isn't too large, as they can be expensive
            if not is_large_dataset:
                compression_types.extend([CompressionType.PREFIX, CompressionType.RLE])
                
        elif column.dtype == DataType.INT:
            compression_types = [CompressionType.RLE, CompressionType.BIT_PACK]
            if not is_large_dataset:
                compression_types.extend([CompressionType.DICT, CompressionType.DELTA])
                
        elif column.dtype in [DataType.FLOAT, DataType.DECIMAL]:
            compression_types = [CompressionType.RLE]
            if not is_large_dataset:
                compression_types.extend([CompressionType.DICT, CompressionType.DELTA])
                
        elif column.dtype == DataType.BOOLEAN:
            # Boolean data benefits most from bit packing
            return CompressionType.BIT_PACK
            
        elif column.dtype in [DataType.TIMESTAMP, DataType.DATE]:
            compression_types = [CompressionType.DELTA, CompressionType.RLE]
            if not is_large_dataset:
                compression_types.append(CompressionType.DICT)
                
        else:
            compression_types = [CompressionType.RLE]
            if not is_large_dataset:
                compression_types.append(CompressionType.DICT)
            
        # Test each compression type
        for comp_type in compression_types:
            # Reset data
            column.data = original_data
            column.compression = comp_type
            column.compressed_data = None
            
            # Apply compression
            try:
                Compression.compress(column)
                ratio = Compression.compression_ratio(column)
                
                # Keep track of the best compression
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_compression = comp_type
                    
            except Exception:
                # If compression fails, skip this algorithm
                continue
        
        # If no significant compression was achieved, use no compression
        if best_compression == CompressionType.NONE or best_ratio > 0.95:
            return CompressionType.NONE
    
        return best_compression
    
    @classmethod
    def detect_best_compression(cls, column) -> CompressionType:
        """
        Detect the best compression algorithm for the column data based on its type and content.
        Uses heuristics to avoid testing all algorithms, which is faster but may be less optimal.
        
        Args:
            column: The column to analyze for compression.
            
        Returns:
            The recommended CompressionType for this column's data.
        """
        if column.data is None or len(column.data) == 0:
            return CompressionType.NONE
            
        # For small datasets, compression might not be worth it
        if len(column.data) < 100:
            return CompressionType.NONE
            
        # For strings, check for prefix commonality vs dictionary benefit
        if column.dtype == DataType.STRING:
            # Check number of unique values
            unique_values = set(column.data[:min(5000, len(column.data))])  # Sample for large datasets
            unique_count = len(unique_values)
            unique_ratio = unique_count / min(5000, len(column.data))
            
            # Estimate average string length
            avg_len = 0
            for val in column.data[:min(1000, len(column.data))]:  # Sample
                avg_len += len(str(val))
            avg_len /= min(1000, len(column.data))
            
            # For very short strings, compression might not be beneficial
            if avg_len < 5 and unique_ratio > 0.2:
                return CompressionType.NONE
            
            # If high repetition, use dictionary encoding
            if unique_ratio < 0.3:  # Increased threshold for unique values
                # Calculate the estimated compression ratio to double-check
                dict_size = 0
                for val in unique_values:
                    dict_size += len(str(val))
                indices_size = len(column.data) * 4  # 4 bytes per index
                
                # Total dictionary size
                total_dict_size = dict_size + indices_size
                
                # Estimate original size (using average length)
                orig_size = len(column.data) * avg_len
                
                # Only use dict compression if it saves space
                if total_dict_size < orig_size * 0.9:  # At least 10% saving
                    return CompressionType.DICT
                else:
                    return CompressionType.NONE
                
            # For strings that might share prefixes (like URLs, file paths)
            sample = column.data[:min(100, len(column.data))]
            prefix_potential = False
            
            for i in range(1, len(sample)):
                prev, curr = str(sample[i-1]), str(sample[i])
                common_prefix_len = 0
                for j in range(min(len(prev), len(curr))):
                    if prev[j] == curr[j]:
                        common_prefix_len += 1
                    else:
                        break
                
                # If significant prefix sharing
                if common_prefix_len > 5 and common_prefix_len > 0.3 * min(len(prev), len(curr)):
                    prefix_potential = True
                    break
            
            if prefix_potential:
                return CompressionType.PREFIX
            
            # If string is very small or unique ratio is high, no compression may be better
            if avg_len < 10 or unique_ratio > 0.5:
                return CompressionType.NONE
                
            # Default to dictionary for other string cases
            return CompressionType.DICT
                
        # For numeric data, check for patterns
        elif column.dtype in [DataType.INT, DataType.FLOAT, DataType.DECIMAL]:
            # Check for runs of identical values
            runs = 0
            for i in range(1, min(1000, len(column.data))):
                if column.data[i] == column.data[i-1]:
                    runs += 1
            run_ratio = runs / min(999, len(column.data) - 1)
            
            # If many runs of identical values, use RLE
            if run_ratio > 0.3:
                return CompressionType.RLE
            
            # For integers with small range, bit packing works well
            if column.dtype == DataType.INT:
                data_range = np.max(column.data) - np.min(column.data)
                if data_range < 2**16:  # If range fits in 16 bits
                    return CompressionType.BIT_PACK
            
            # For sequential or near-sequential data, delta encoding
            if len(column.data) > 2:
                # Sample the data to check if it's sequential
                diffs = np.diff(column.data[:min(1000, len(column.data))])
                avg_diff = np.mean(np.abs(diffs))
                max_diff = np.max(np.abs(diffs))
                
                if max_diff < 100 * avg_diff:  # Not too many large jumps
                    return CompressionType.DELTA
            
            # Default to dictionary for numeric data
            return CompressionType.DICT
            
        # For boolean data
        elif column.dtype == DataType.BOOLEAN:
            # Bit packing is efficient for boolean data
            return CompressionType.BIT_PACK
            
        # For timestamps and dates
        elif column.dtype in [DataType.TIMESTAMP, DataType.DATE]:
            # Check for sequential times
            if len(column.data) > 2:
                # Try delta encoding for timestamps/dates since they're often sequential
                return CompressionType.DELTA
                
        # Default fallback
        return CompressionType.DICT
    
    @classmethod
    def _run_length_encode(cls, column) -> Dict[str, Any]:
        """
        Run-length encode the column data.
        
        Returns:
            A dictionary containing the encoded values and counts.
        """
        if column.data is None:
            return {"values": [], "counts": []}
            
        data = column.data
        if len(data) == 0:
            return {"values": [], "counts": []}
        
        values = []
        counts = []
        
        current_value = data[0]
        current_count = 1
        
        for i in range(1, len(data)):
            if data[i] == current_value:
                current_count += 1
            else:
                values.append(current_value)
                counts.append(current_count)
                current_value = data[i]
                current_count = 1
        
        values.append(current_value)
        counts.append(current_count)
        
        return {
            "values": np.array(values, dtype=column.numpy_dtype),
            "counts": np.array(counts, dtype=np.int32)
        }
    
    @classmethod
    def _run_length_decode(cls, column) -> np.ndarray:
        """
        Decode run-length encoded data.
        
        Returns:
            The decoded data as a NumPy array.
        """
        if column.compressed_data is None:
            return np.array([], dtype=column.numpy_dtype)
            
        values = column.compressed_data.get("values", np.array([], dtype=column.numpy_dtype))
        counts = column.compressed_data.get("counts", np.array([], dtype=np.int32))
        
        if len(values) == 0 or len(counts) == 0:
            return np.array([], dtype=column.numpy_dtype)
        
        result = np.empty(int(np.sum(counts)), dtype=column.numpy_dtype)
        
        pos = 0
        for i in range(len(values)):
            value = values[i]
            count = counts[i]
            result[pos:pos+count] = value
            pos += count
        
        return result
    
    @classmethod
    def _dictionary_encode(cls, column) -> Dict[str, Any]:
        """
        Dictionary encode the column data.
        
        Args:
            column: The column to encode.
            
        Returns:
            A dictionary containing the unique values and indices.
        """
        if column.data is None:
            return {"values": [], "indices": []}
            
        data = column.data
        if len(data) == 0:
            return {"values": [], "indices": []}
        
        # For string columns, perform a size check before applying dictionary encoding
        if column.dtype == DataType.STRING:
            # Count unique values
            unique_count = len(set(data))
            total_count = len(data)
            
            # If the number of unique values is too high compared to total values,
            # dictionary encoding might not be efficient
            if unique_count > total_count * 0.8:  # More than 80% unique
                return {"values": data, "indices": np.arange(len(data)), "no_compression": True}
        
        # Get unique values
        unique_values, indices = np.unique(data, return_inverse=True)
        
        # For string data, ensure we're getting compression benefit
        if column.dtype == DataType.STRING:
            # Calculate estimated size of dictionary representation
            values_size = 0
            for val in unique_values:
                values_size += len(str(val))
            
            # Size of indices (assuming 32-bit integers)
            indices_size = len(indices) * 4
            
            # Total dictionary size
            dict_size = values_size + indices_size
            
            # Original data size
            orig_size = sum(len(str(val)) for val in data)
            
            # If dictionary encoding doesn't save space, don't use it
            if dict_size >= orig_size:
                return {"values": data, "indices": np.arange(len(data)), "no_compression": True}
        
        return {
            "values": unique_values,
            "indices": indices
        }
    
    @classmethod
    def _dictionary_decode(cls, column) -> np.ndarray:
        """
        Decode dictionary encoded data.
        
        Returns:
            The decoded data as a NumPy array.
        """
        if column.compressed_data is None:
            return np.array([], dtype=column.numpy_dtype)
            
        values = column.compressed_data.get("values", np.array([], dtype=column.numpy_dtype))
        indices = column.compressed_data.get("indices", np.array([], dtype=np.int32))
        no_compression = column.compressed_data.get("no_compression", False)
        
        if len(values) == 0 or len(indices) == 0:
            return np.array([], dtype=column.numpy_dtype)
        
        # If we marked this as not worth compressing, just return the original values
        if no_compression:
            return values
            
        return values[indices]
    
    @classmethod
    def _delta_encode(cls, column) -> Dict[str, Any]:
        """
        Delta encode the column data (for numeric types only).
        
        Args:
            column: The column to encode.
            
        Returns:
            A dictionary containing the base value and deltas.
        """
        if column.data is None or len(column.data) == 0:
            return {"base": None, "deltas": []}
        
        # Only applicable for numeric types
        if column.dtype not in [DataType.INT, DataType.FLOAT, DataType.DECIMAL]:
            # Fall back to dictionary encoding for non-numeric types
            return cls._dictionary_encode(column)
        
        data = column.data
        base = data[0]
        if len(data) == 1:
            return {"base": base, "deltas": np.array([], dtype=np.int32)}
        
        deltas = np.diff(data)
        return {"base": base, "deltas": deltas}
    
    @classmethod
    def _delta_decode(cls, column) -> np.ndarray:
        """
        Decode delta encoded data.
        
        Returns:
            The decoded data as a NumPy array.
        """
        if column.compressed_data is None:
            return np.array([], dtype=column.numpy_dtype)
        
        base = column.compressed_data.get("base")
        deltas = column.compressed_data.get("deltas", np.array([], dtype=column.numpy_dtype))
        
        if base is None:
            return np.array([], dtype=column.numpy_dtype)
        
        if len(deltas) == 0:
            return np.array([base], dtype=column.numpy_dtype)
        
        result = np.empty(len(deltas) + 1, dtype=column.numpy_dtype)
        result[0] = base
        
        # Cumulative sum to reconstruct original values
        np.cumsum(np.concatenate(([base], deltas)), out=result)
        
        return result
    
    @classmethod
    def _bit_pack_encode(cls, column) -> Dict[str, Any]:
        """
        Bit-pack encoding for integer data.
        
        Args:
            column: The column to encode.
            
        Returns:
            A dictionary containing the packed data and metadata.
        """
        if column.data is None or len(column.data) == 0:
            return {"packed": bytearray(), "num_values": 0, "bit_width": 0}
        
        # Only applicable for integer types
        if column.dtype != DataType.INT:
            # Fall back to dictionary encoding
            return cls._dictionary_encode(column)
        
        data = column.data
        
        # Calculate bit width needed
        if len(data) == 0:
            return {"packed": bytearray(), "num_values": 0, "bit_width": 0}
        
        min_val = np.min(data)
        max_val = np.max(data)
        
        # If all values are the same, special case
        if min_val == max_val:
            return {"packed": bytearray(), "num_values": len(data), "bit_width": 0, "constant": min_val}
        
        # Calculate range and required bits
        value_range = max_val - min_val
        bit_width = max(1, int(np.ceil(np.log2(value_range + 1))))
        
        # Normalize values to start from 0
        normalized = data - min_val
        
        # Pack bits
        packed_bytes = bytearray()
        current_byte = 0
        bits_filled = 0
        
        for value in normalized:
            value_int = int(value)
            bits_remaining = bit_width
            
            while bits_remaining > 0:
                bits_to_pack = min(8 - bits_filled, bits_remaining)
                bits_value = (value_int >> (bits_remaining - bits_to_pack)) & ((1 << bits_to_pack) - 1)
                current_byte = (current_byte << bits_to_pack) | bits_value
                bits_filled += bits_to_pack
                bits_remaining -= bits_to_pack
                
                if bits_filled == 8:
                    packed_bytes.append(current_byte)
                    current_byte = 0
                    bits_filled = 0
        
        # Add remaining bits if any
        if bits_filled > 0:
            current_byte <<= (8 - bits_filled)
            packed_bytes.append(current_byte)
        
        return {
            "packed": packed_bytes,
            "num_values": len(data),
            "bit_width": bit_width,
            "min_val": min_val
        }
    
    @classmethod
    def _bit_pack_decode(cls, column) -> np.ndarray:
        """
        Decode bit-packed data.
        
        Args:
            column: The column to decode.
            
        Returns:
            The decoded data as a NumPy array.
        """
        if column.compressed_data is None:
            return np.array([], dtype=column.numpy_dtype)
        
        packed = column.compressed_data.get("packed", bytearray())
        num_values = column.compressed_data.get("num_values", 0)
        bit_width = column.compressed_data.get("bit_width", 0)
        min_val = column.compressed_data.get("min_val", 0)
        constant = column.compressed_data.get("constant")
        
        # Special case for constant value
        if constant is not None:
            return np.full(num_values, constant, dtype=column.numpy_dtype)
        
        if num_values == 0 or bit_width == 0:
            return np.array([], dtype=column.numpy_dtype)
        
        result = np.empty(num_values, dtype=column.numpy_dtype)
        
        # Unpack bits
        value_index = 0
        bit_position = 0
        
        for i in range(num_values):
            value = 0
            bits_read = 0
            
            while bits_read < bit_width:
                bits_to_read = min(8 - bit_position, bit_width - bits_read)
                byte = packed[value_index] if value_index < len(packed) else 0
                
                # Extract bits from current byte
                mask = ((1 << bits_to_read) - 1) << (8 - bit_position - bits_to_read)
                extracted = (byte & mask) >> (8 - bit_position - bits_to_read)
                
                # Add to value
                value = (value << bits_to_read) | extracted
                
                bits_read += bits_to_read
                bit_position += bits_to_read
                
                if bit_position == 8:
                    value_index += 1
                    bit_position = 0
            
            result[i] = value + min_val
        
        return result
    
    @classmethod
    def _prefix_encode(cls, column) -> Dict[str, Any]:
        """
        Prefix encoding for string data.
        
        Args:
            column: The column to encode.
            
        Returns:
            A dictionary containing the encoded data.
        """
        if column.data is None or len(column.data) == 0:
            return {"prefixes": [], "suffixes": []}
        
        # Only applicable for string type
        if column.dtype != DataType.STRING:
            # Fall back to dictionary encoding
            return cls._dictionary_encode(column)
        
        data = column.data
        
        # Sort data to maximize prefix sharing
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        
        prefixes = []
        suffixes = []
        prefix_lens = []
        
        prev_str = ""
        for curr_str in sorted_data:
            # Find common prefix length
            prefix_len = 0
            max_len = min(len(prev_str), len(curr_str))
            
            while prefix_len < max_len and prev_str[prefix_len] == curr_str[prefix_len]:
                prefix_len += 1
            
            prefixes.append(prefix_len)
            suffixes.append(curr_str[prefix_len:])
            prev_str = curr_str
        
        # Store original indices for reconstruction
        return {
            "prefixes": np.array(prefixes, dtype=np.int32),
            "suffixes": np.array(suffixes),
            "sorted_indices": sorted_indices
        }
    
    @classmethod
    def _prefix_decode(cls, column) -> np.ndarray:
        """
        Decode prefix-encoded data.
        
        Args:
            column: The column to decode.
            
        Returns:
            The decoded data as a NumPy array.
        """
        if column.compressed_data is None:
            return np.array([], dtype=column.numpy_dtype)
        
        prefixes = column.compressed_data.get("prefixes", [])
        suffixes = column.compressed_data.get("suffixes", [])
        sorted_indices = column.compressed_data.get("sorted_indices", [])
        
        if len(prefixes) == 0 or len(suffixes) == 0 or len(sorted_indices) == 0:
            return np.array([], dtype=column.numpy_dtype)
        
        # Reconstruct original strings
        sorted_result = np.empty(len(prefixes), dtype=object)
        
        prev_str = ""
        for i, (prefix_len, suffix) in enumerate(zip(prefixes, suffixes)):
            if prefix_len == 0:
                sorted_result[i] = suffix
            else:
                sorted_result[i] = prev_str[:prefix_len] + suffix
            
            prev_str = sorted_result[i]
        
        # Restore original order
        result = np.empty(len(sorted_result), dtype=object)
        for i, orig_idx in enumerate(sorted_indices):
            result[orig_idx] = sorted_result[i]
        
        return result.astype(column.numpy_dtype)
    
    @classmethod
    def _zlib_encode(cls, column) -> Dict[str, Any]:
        """
        Compress data using zlib (fallback for ZSTD and LZ4).
        
        Args:
            column: The column to encode.
            
        Returns:
            A dictionary containing the compressed data.
        """
        if column.data is None or len(column.data) == 0:
            return {"compressed": b'', "original_type": str(column.dtype)}
        
        import zlib
        import pickle
        
        # Serialize the data
        serialized = pickle.dumps(column.data)
        
        # Compress with zlib
        compressed = zlib.compress(serialized)
        
        return {
            "compressed": compressed,
            "original_type": str(column.dtype)
        }
    
    @classmethod
    def _zlib_decode(cls, column) -> np.ndarray:
        """
        Decompress zlib-compressed data.
        
        Returns:
            The decompressed data as a NumPy array.
        """
        if column.compressed_data is None:
            return np.array([], dtype=column.numpy_dtype)
        
        compressed = column.compressed_data.get("compressed", b'')
        
        if not compressed:
            return np.array([], dtype=column.numpy_dtype)
        
        import zlib
        import pickle
        
        # Decompress and deserialize
        decompressed = zlib.decompress(compressed)
        result = pickle.loads(decompressed)
        
        return result
    