import struct
import pickle
from typing import Tuple
from .column import Column, DataType, CompressionType

class Deserializer:
    """Base class for deserializers."""
    
    
    @classmethod
    def deserialize(cls, data: bytes) -> Tuple['Column', int]:
        """
        Deserialize a column from bytes.
        
        Args:
            data: The serialized column data.
            
        Returns:
            A tuple of (deserialized column, number of bytes consumed).
        """
        # Parse header
        name_length = struct.unpack('!I', data[:4])[0]
        name = data[4:4+name_length].decode('utf-8')
        dtype_value, compression_value = struct.unpack('!II', data[4+name_length:12+name_length])
        
        dtype = DataType(dtype_value)
        compression = CompressionType(compression_value)
        
        offset = 12 + name_length
        
        # Create column
        column = Column(name, dtype, compression=compression)
        
        # Parse metadata (min/max values)
        if dtype == DataType.INT:
            column.min_value, column.max_value = struct.unpack('!qq', data[offset:offset+16])
            offset += 16
        elif dtype == DataType.FLOAT:
            column.min_value, column.max_value = struct.unpack('!dd', data[offset:offset+16])
            offset += 16
        elif dtype == DataType.BOOLEAN:
            column.min_value, column.max_value = struct.unpack('!??', data[offset:offset+2])
            offset += 2
        elif dtype == DataType.TIMESTAMP:
            column.min_value, column.max_value = struct.unpack('!qq', data[offset:offset+16])
            offset += 16
        elif dtype == DataType.DECIMAL:
            min_length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            column.min_value = data[offset:offset+min_length].decode('utf-8')
            offset += min_length
            
            max_length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            column.max_value = data[offset:offset+max_length].decode('utf-8')
            offset += max_length
        elif dtype == DataType.DATE:
            column.min_value, column.max_value = struct.unpack('!qq', data[offset:offset+16])
            offset += 16
        elif dtype == DataType.STRING:
            min_length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            column.min_value = data[offset:offset+min_length].decode('utf-8')
            offset += min_length
            
            max_length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            column.max_value = data[offset:offset+max_length].decode('utf-8')
            offset += max_length
        
        # Parse data
        import pickle
        
        # Read the compression type (this matches the format in serialize.py)
        try:
            compression_value = struct.unpack('!I', data[offset:offset+4])[0]
            compression = CompressionType(compression_value)
            offset += 4
        except (struct.error, ValueError):
            # If we can't read the compression type, use the one from the header
            pass
        
        if compression == CompressionType.RLE:
            # Read RLE compressed data format
            values_size, counts_size = struct.unpack('!II', data[offset:offset+8])
            offset += 8
            
            if values_size > 0 and counts_size > 0:
                values = pickle.loads(data[offset:offset+values_size])
                offset += values_size
                counts = pickle.loads(data[offset:offset+counts_size])
                offset += counts_size
                
                column.compressed_data = {
                    "values": values,
                    "counts": counts
                }
        
        elif compression == CompressionType.DICT:
            # Read dictionary compressed data format
            values_size, indices_size = struct.unpack('!II', data[offset:offset+8])
            offset += 8
            
            if values_size > 0 and indices_size > 0:
                values = pickle.loads(data[offset:offset+values_size])
                offset += values_size
                indices = pickle.loads(data[offset:offset+indices_size])
                offset += indices_size
                
                column.compressed_data = {
                    "values": values,
                    "indices": indices
                }
                
        elif compression == CompressionType.DELTA:
            # Read delta compressed data format
            base_size, deltas_size = struct.unpack('!II', data[offset:offset+8])
            offset += 8
            
            base = pickle.loads(data[offset:offset+base_size])
            offset += base_size
            deltas = pickle.loads(data[offset:offset+deltas_size])
            offset += deltas_size
            
            column.compressed_data = {
                "base": base,
                "deltas": deltas
            }
            
        elif compression == CompressionType.BIT_PACK:
            # Read bit-pack compressed data format
            packed_size, has_constant, num_values, bit_width = struct.unpack('!I?II', data[offset:offset+13])
            offset += 13
            
            packed = data[offset:offset+packed_size]
            offset += packed_size
            
            compressed_data = {
                "packed": packed,
                "num_values": num_values,
                "bit_width": bit_width
            }
            
            if has_constant:
                constant_size = struct.unpack('!I', data[offset:offset+4])[0]
                offset += 4
                constant = pickle.loads(data[offset:offset+constant_size])
                offset += constant_size
                compressed_data["constant"] = constant
            
            min_val_size = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            min_val = pickle.loads(data[offset:offset+min_val_size])
            offset += min_val_size
            
            compressed_data["min_val"] = min_val
            column.compressed_data = compressed_data
            
        elif compression == CompressionType.PREFIX:
            # Read prefix compressed data format
            prefixes_size, suffixes_size, indices_size = struct.unpack('!III', data[offset:offset+12])
            offset += 12
            
            prefixes = pickle.loads(data[offset:offset+prefixes_size])
            offset += prefixes_size
            
            suffixes = pickle.loads(data[offset:offset+suffixes_size])
            offset += suffixes_size
            
            sorted_indices = pickle.loads(data[offset:offset+indices_size])
            offset += indices_size
            
            column.compressed_data = {
                "prefixes": prefixes,
                "suffixes": suffixes,
                "sorted_indices": sorted_indices
            }
            
        elif compression == CompressionType.ZSTD or compression == CompressionType.LZ4:
            # Read zlib compressed data format
            compressed_size, type_size = struct.unpack('!II', data[offset:offset+8])
            offset += 8
            
            original_type = data[offset:offset+type_size].decode('utf-8')
            offset += type_size
            
            compressed = data[offset:offset+compressed_size]
            offset += compressed_size
            
            column.compressed_data = {
                "compressed": compressed,
                "original_type": original_type
            }
        
        # Decompress data if we have compressed data
        if column.compressed_data is not None:
            column.data = column.decompress()
        
        else:  # NONE (uncompressed)
            # Parse raw data
            data_length = struct.unpack('!I', data[offset:offset+4])[0]
            offset += 4
            
            raw_data = []
            for _ in range(data_length):
                if dtype == DataType.INT:
                    value = struct.unpack('!q', data[offset:offset+8])[0]
                    offset += 8
                elif dtype == DataType.FLOAT:
                    value = struct.unpack('!d', data[offset:offset+8])[0]
                    offset += 8
                elif dtype == DataType.BOOLEAN:
                    value = struct.unpack('!?', data[offset:offset+1])[0]
                    offset += 1
                elif dtype == DataType.TIMESTAMP:
                    value = struct.unpack('!q', data[offset:offset+8])[0]
                    offset += 8
                elif dtype == DataType.DECIMAL:
                    value_length = struct.unpack('!I', data[offset:offset+4])[0]
                    offset += 4
                    value = data[offset:offset+value_length].decode('utf-8')
                    offset += value_length
                    # Convert string representation back to decimal
                    from decimal import Decimal
                    value = Decimal(value)
                elif dtype == DataType.DATE:
                    value = struct.unpack('!q', data[offset:offset+8])[0]
                    offset += 8
                else:  # STRING and other types
                    value_length = struct.unpack('!I', data[offset:offset+4])[0]
                    offset += 4
                    value = data[offset:offset+value_length].decode('utf-8')
                    offset += value_length
                
                raw_data.append(value)
            
            column.set_data(raw_data)
        
        return column, offset
