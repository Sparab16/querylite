import struct
import pickle

import struct
import pickle
from .column import DataType, CompressionType

class Serializer:
    """Base class for serializers."""
    
    @classmethod
    def serialize(cls, column) -> bytes:
        """
        Serialize the column to bytes.
        
        Returns:
            The serialized column as bytes.
        """
        import pickle
        
        # Header: name length (4 bytes) + name + dtype (4 bytes) + compression (4 bytes)
        name_bytes = column.name.encode('utf-8')
        header = struct.pack(f'!I{len(name_bytes)}sII', 
                             len(name_bytes), 
                             name_bytes, 
                             column.dtype.value, 
                             column.compression.value)
        
        # Metadata: min and max values (for indexing)
        if column.min_value is not None and column.max_value is not None:
            if column.dtype == DataType.INT:
                metadata = struct.pack('!qq', int(column.min_value), int(column.max_value))
            elif column.dtype == DataType.FLOAT:
                metadata = struct.pack('!dd', float(column.min_value), float(column.max_value))
            elif column.dtype == DataType.BOOLEAN:
                # For boolean, just store as bytes (1 for True, 0 for False)
                metadata = struct.pack('!??', bool(column.min_value), bool(column.max_value))
            elif column.dtype == DataType.TIMESTAMP:
                # For timestamp, store as int64 (unix timestamp in nanoseconds)
                metadata = struct.pack('!qq', int(column.min_value), int(column.max_value))
            elif column.dtype == DataType.DECIMAL:
                # For decimal, store as string representation to preserve precision
                min_bytes = str(column.min_value).encode('utf-8')
                max_bytes = str(column.max_value).encode('utf-8')
                metadata = struct.pack(f'!I{len(min_bytes)}sI{len(max_bytes)}s',
                                      len(min_bytes), min_bytes,
                                      len(max_bytes), max_bytes)
            elif column.dtype == DataType.DATE:
                # For date, store as int (days since epoch)
                metadata = struct.pack('!qq', int(column.min_value), int(column.max_value))
            else:  # STRING and other types
                min_bytes = str(column.min_value).encode('utf-8')
                max_bytes = str(column.max_value).encode('utf-8')
                metadata = struct.pack(f'!I{len(min_bytes)}sI{len(max_bytes)}s',
                                      len(min_bytes), min_bytes,
                                      len(max_bytes), max_bytes)
        else:
            metadata = b''
        
        # Data
        data_bytes = b''
        if column.compression != CompressionType.NONE and column.compressed_data is not None:
            # First, store the compression type
            data_bytes = struct.pack('!I', column.compression.value)
            
            # Convert compressed data to bytes using JSON representation for most formats
            if column.compression == CompressionType.RLE:
                values = column.compressed_data.get("values")
                counts = column.compressed_data.get("counts")
                
                # Pack values and counts arrays
                if values is not None and counts is not None:
                    values_bytes = pickle.dumps(values)
                    counts_bytes = pickle.dumps(counts)
                    data_bytes += struct.pack('!II', len(values_bytes), len(counts_bytes))
                    data_bytes += values_bytes + counts_bytes
                else:
                    data_bytes += struct.pack('!II', 0, 0)
                    
            elif column.compression == CompressionType.DICT:
                values = column.compressed_data.get("values")
                indices = column.compressed_data.get("indices")
                
                if values is not None and indices is not None:
                    values_bytes = pickle.dumps(values)
                    indices_bytes = pickle.dumps(indices)
                    data_bytes += struct.pack('!II', len(values_bytes), len(indices_bytes))
                    data_bytes += values_bytes + indices_bytes
                else:
                    data_bytes += struct.pack('!II', 0, 0)
                    
            elif column.compression == CompressionType.DELTA:
                base = column.compressed_data.get("base")
                deltas = column.compressed_data.get("deltas")
                
                base_bytes = pickle.dumps(base)
                deltas_bytes = pickle.dumps(deltas)
                data_bytes += struct.pack('!II', len(base_bytes), len(deltas_bytes))
                data_bytes += base_bytes + deltas_bytes
                
            elif column.compression == CompressionType.BIT_PACK:
                # Serialize all bit-pack metadata fields
                packed_bytes = column.compressed_data.get("packed", bytearray())
                num_values = column.compressed_data.get("num_values", 0)
                bit_width = column.compressed_data.get("bit_width", 0)
                min_val = column.compressed_data.get("min_val", 0)
                constant = column.compressed_data.get("constant")
                
                has_constant = constant is not None
                data_bytes += struct.pack('!I?II', len(packed_bytes), has_constant, num_values, bit_width)
                data_bytes += packed_bytes
                
                if has_constant:
                    constant_bytes = pickle.dumps(constant)
                    data_bytes += struct.pack('!I', len(constant_bytes))
                    data_bytes += constant_bytes
                
                min_val_bytes = pickle.dumps(min_val)
                data_bytes += struct.pack('!I', len(min_val_bytes))
                data_bytes += min_val_bytes
                
            elif column.compression == CompressionType.PREFIX:
                prefixes = column.compressed_data.get("prefixes")
                suffixes = column.compressed_data.get("suffixes")
                sorted_indices = column.compressed_data.get("sorted_indices")
                
                prefixes_bytes = pickle.dumps(prefixes)
                suffixes_bytes = pickle.dumps(suffixes)
                sorted_indices_bytes = pickle.dumps(sorted_indices)
                
                data_bytes += struct.pack('!III', 
                                         len(prefixes_bytes), 
                                         len(suffixes_bytes),
                                         len(sorted_indices_bytes))
                data_bytes += prefixes_bytes + suffixes_bytes + sorted_indices_bytes
                
            elif column.compression == CompressionType.ZSTD or column.compression == CompressionType.LZ4:
                # For ZSTD/LZ4, we use zlib compression
                compressed = column.compressed_data.get("compressed", b'')
                original_type = column.compressed_data.get("original_type", "")
                
                type_bytes = original_type.encode('utf-8')
                data_bytes += struct.pack(f'!II{len(type_bytes)}s', 
                                         len(compressed), 
                                         len(type_bytes),
                                         type_bytes)
                data_bytes += compressed
        elif column.data is not None:
            # Uncompressed data
            data_bytes = struct.pack('!I', len(column.data))
            for value in column.data:
                if column.dtype == DataType.INT:
                    data_bytes += struct.pack('!q', int(value))
                elif column.dtype == DataType.FLOAT:
                    data_bytes += struct.pack('!d', float(value))
                elif column.dtype == DataType.BOOLEAN:
                    data_bytes += struct.pack('!?', bool(value))
                elif column.dtype == DataType.TIMESTAMP:
                    # Store timestamp as int64 (nanoseconds since epoch)
                    data_bytes += struct.pack('!q', int(value))
                elif column.dtype == DataType.DECIMAL:
                    # Store decimal as string to preserve precision
                    value_bytes = str(value).encode('utf-8')
                    data_bytes += struct.pack(f'!I{len(value_bytes)}s',
                                           len(value_bytes), value_bytes)
                elif column.dtype == DataType.DATE:
                    # Store date as int64 (days since epoch)
                    data_bytes += struct.pack('!q', int(value))
                else:  # STRING and other types
                    value_bytes = str(value).encode('utf-8')
                    data_bytes += struct.pack(f'!I{len(value_bytes)}s',
                                           len(value_bytes), value_bytes)
        else:
            # No data
            data_bytes = struct.pack('!I', 0)  # data size = 0
        
        return header + metadata + data_bytes
    
