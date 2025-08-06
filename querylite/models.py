from enum import Enum

class DataType(Enum):
    """Enum representing supported data types."""
    INT = 1       # 64-bit integer
    FLOAT = 2     # 64-bit floating point
    STRING = 3    # Variable-length string
    BOOLEAN = 4   # Boolean value
    TIMESTAMP = 5 # Datetime timestamp
    DECIMAL = 8   # Fixed precision decimal
    DATE = 9      # Date only


class CompressionType(Enum):
    """Enum representing supported compression algorithms."""
    NONE = 0      # No compression
    RLE = 1       # Run-Length Encoding
    DICT = 2      # Dictionary Encoding
    DELTA = 3     # Delta encoding (for sequences)
    BIT_PACK = 4  # Bit packing (for integers)
    PREFIX = 5    # Common prefix compression (for strings)
    ZSTD = 6      # Zstandard compression (general purpose)
    LZ4 = 7       # LZ4 compression (fast general purpose)
    AUTO = 8      # Automatically select the best compression algorithm
