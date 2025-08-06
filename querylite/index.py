"""
Index module for handling column-level indexing to speed up queries.
"""
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import numpy as np
from enum import Enum
import struct
import pickle


class IndexType(Enum):
    """Types of indices supported."""
    MINMAX = 1  # Min-Max index for range queries
    BITMAP = 2  # Bitmap index for equality queries
    DICTIONARY = 3  # Dictionary index for string lookups


class Index:
    """
    Base class for column indices.
    
    Attributes:
        column_name: The name of the column this index is for.
        index_type: The type of the index.
    """
    
    def __init__(self, column_name: str, index_type: IndexType):
        """
        Initialize an index.
        
        Args:
            column_name: The name of the column this index is for.
            index_type: The type of the index.
        """
        self.column_name = column_name
        self.index_type = index_type
    
    def can_satisfy(self, operator: str, value: Any) -> bool:
        """
        Check if this index can be used to satisfy a query with the given operator and value.
        
        Args:
            operator: The comparison operator.
            value: The value to compare against.
            
        Returns:
            True if the index can be used to satisfy the query.
        """
        return False
    
    def evaluate(self, operator: str, value: Any) -> List[int]:
        """
        Use the index to evaluate a query.
        
        Args:
            operator: The comparison operator.
            value: The value to compare against.
            
        Returns:
            A list of row indices that satisfy the query.
        """
        return []
    
    def serialize(self) -> bytes:
        """
        Serialize the index to bytes.
        
        Returns:
            The serialized index as bytes.
        """
        header = struct.pack('!BI', self.index_type.value, len(self.column_name))
        return header + self.column_name.encode('utf-8')
    
    @classmethod
    def deserialize(cls, data: bytes) -> Tuple['Index', int]:
        """
        Deserialize an index from bytes.
        
        Args:
            data: The serialized index.
            
        Returns:
            A tuple of (index, bytes consumed).
        """
        index_type_val, name_len = struct.unpack('!BI', data[:5])
        index_type = IndexType(index_type_val)
        column_name = data[5:5+name_len].decode('utf-8')
        
        if index_type == IndexType.MINMAX:
            return MinMaxIndex.deserialize_content(column_name, data[5+name_len:])
        elif index_type == IndexType.BITMAP:
            return BitmapIndex.deserialize_content(column_name, data[5+name_len:])
        elif index_type == IndexType.DICTIONARY:
            return DictionaryIndex.deserialize_content(column_name, data[5+name_len:])
        
        # Default if we don't recognize the index type
        return cls(column_name, index_type), 5 + name_len


class MinMaxIndex(Index):
    """
    Min-Max index for range queries. Stores the min and max values for a column.
    Useful for quickly determining if a range query can be satisfied.
    """
    
    def __init__(self, column_name: str, min_value: Any, max_value: Any):
        """
        Initialize a min-max index.
        
        Args:
            column_name: The name of the column this index is for.
            min_value: The minimum value in the column.
            max_value: The maximum value in the column.
        """
        super().__init__(column_name, IndexType.MINMAX)
        self.min_value = min_value
        self.max_value = max_value
    
    def can_satisfy(self, operator: str, value: Any) -> bool:
        """
        Check if this index can be used to satisfy a query.
        
        Args:
            operator: The comparison operator.
            value: The value to compare against.
            
        Returns:
            True if the index can be used to satisfy the query.
        """
        if self.min_value is None or self.max_value is None:
            return False
        
        try:
            # For range queries
            if operator == ">":
                return value < self.max_value  # Some values might be greater than value
            elif operator == ">=":
                return value <= self.max_value
            elif operator == "<":
                return value > self.min_value  # Some values might be less than value
            elif operator == "<=":
                return value >= self.min_value
            elif operator == "=" or operator == "==":
                return self.min_value <= value <= self.max_value
            elif operator == "!=" or operator == "<>":
                return True  # Can't optimize not equals with min-max
        except Exception:
            # If comparison fails, we can't satisfy
            return False
        
        return False
    
    def evaluate(self, operator: str, value: Any) -> List[int]:
        """
        Min-Max index can't directly evaluate queries, it can only determine
        if a full scan is needed. Returns empty list to indicate full scan is required.
        """
        return []
    
    def serialize(self) -> bytes:
        """
        Serialize the index to bytes.
        
        Returns:
            The serialized index as bytes.
        """
        base_data = super().serialize()
        
        # Serialize min and max values
        content = pickle.dumps((self.min_value, self.max_value))
        content_len = struct.pack('!I', len(content))
        
        return base_data + content_len + content
    
    @classmethod
    def deserialize_content(cls, column_name: str, data: bytes) -> Tuple['MinMaxIndex', int]:
        """
        Deserialize a min-max index from bytes.
        
        Args:
            column_name: The name of the column this index is for.
            data: The serialized index content.
            
        Returns:
            A tuple of (index, bytes consumed).
        """
        content_len = struct.unpack('!I', data[:4])[0]
        content = data[4:4+content_len]
        min_value, max_value = pickle.loads(content)
        
        return cls(column_name, min_value, max_value), 4 + content_len


class BitmapIndex(Index):
    """
    Bitmap index for equality queries. Stores a bitmap for each unique value.
    Useful for quickly finding rows that match a specific value.
    """
    
    def __init__(self, column_name: str, value_to_bitmap: Optional[Dict[Any, np.ndarray]] = None):
        """
        Initialize a bitmap index.
        
        Args:
            column_name: The name of the column this index is for.
            value_to_bitmap: A dictionary mapping values to bitmaps.
        """
        super().__init__(column_name, IndexType.BITMAP)
        self.value_to_bitmap: Dict[Any, np.ndarray] = {} if value_to_bitmap is None else value_to_bitmap
    
    def add_value(self, value: Any, row_idx: int, total_rows: int) -> None:
        """
        Add a value to the index.
        
        Args:
            value: The value to add.
            row_idx: The row index where the value appears.
            total_rows: The total number of rows in the table.
        """
        if value not in self.value_to_bitmap:
            self.value_to_bitmap[value] = np.zeros(total_rows, dtype=bool)
        
        self.value_to_bitmap[value][row_idx] = True
    
    def can_satisfy(self, operator: str, value: Any) -> bool:
        """
        Check if this index can be used to satisfy a query.
        
        Args:
            operator: The comparison operator.
            value: The value to compare against.
            
        Returns:
            True if the index can be used to satisfy the query.
        """
        # Bitmap index is best for equality
        if operator == "=" or operator == "==":
            return value in self.value_to_bitmap
        
        # For not equals, we can use the index if we have all values
        if (operator == "!=" or operator == "<>") and value in self.value_to_bitmap:
            return True
        
        return False
    
    def evaluate(self, operator: str, value: Any) -> List[int]:
        """
        Use the index to evaluate a query.
        
        Args:
            operator: The comparison operator.
            value: The value to compare against.
            
        Returns:
            A list of row indices that satisfy the query.
        """
        if not self.can_satisfy(operator, value):
            return []
        
        if operator == "=" or operator == "==":
            # For equality, return the bitmap directly
            return np.where(self.value_to_bitmap[value])[0].tolist()
        elif operator == "!=" or operator == "<>":
            # For not equals, invert the bitmap
            return np.where(~self.value_to_bitmap[value])[0].tolist()
        
        return []
    
    def serialize(self) -> bytes:
        """
        Serialize the index to bytes.
        
        Returns:
            The serialized index as bytes.
        """
        base_data = super().serialize()
        
        # Serialize the bitmap dictionary
        content = pickle.dumps(self.value_to_bitmap)
        content_len = struct.pack('!I', len(content))
        
        return base_data + content_len + content
    
    @classmethod
    def deserialize_content(cls, column_name: str, data: bytes) -> Tuple['BitmapIndex', int]:
        """
        Deserialize a bitmap index from bytes.
        
        Args:
            column_name: The name of the column this index is for.
            data: The serialized index content.
            
        Returns:
            A tuple of (index, bytes consumed).
        """
        content_len = struct.unpack('!I', data[:4])[0]
        content = data[4:4+content_len]
        value_to_bitmap = pickle.loads(content)
        
        return cls(column_name, value_to_bitmap), 4 + content_len


class DictionaryIndex(Index):
    """
    Dictionary index for string lookups. Maps string values to row indices.
    Useful for quickly finding rows that match a specific string pattern.
    """
    
    def __init__(self, column_name: str, value_to_rows: Optional[Dict[str, List[int]]] = None):
        """
        Initialize a dictionary index.
        
        Args:
            column_name: The name of the column this index is for.
            value_to_rows: A dictionary mapping string values to row indices.
        """
        super().__init__(column_name, IndexType.DICTIONARY)
        self.value_to_rows: Dict[str, List[int]] = {} if value_to_rows is None else value_to_rows
    
    def add_value(self, value: str, row_idx: int) -> None:
        """
        Add a value to the index.
        
        Args:
            value: The value to add.
            row_idx: The row index where the value appears.
        """
        if value not in self.value_to_rows:
            self.value_to_rows[value] = []
        
        self.value_to_rows[value].append(row_idx)
    
    def can_satisfy(self, operator: str, value: Any) -> bool:
        """
        Check if this index can be used to satisfy a query.
        
        Args:
            operator: The comparison operator.
            value: The value to compare against.
            
        Returns:
            True if the index can be used to satisfy the query.
        """
        # Dictionary index is good for equality and pattern matching
        if operator == "=" or operator == "==":
            return value in self.value_to_rows
        elif operator == "LIKE":
            # For LIKE with a prefix pattern, we can use the index
            if isinstance(value, str) and value.endswith('%') and not value.startswith('%'):
                prefix = value[:-1]
                return any(k.startswith(prefix) for k in self.value_to_rows.keys())
        
        return False
    
    def evaluate(self, operator: str, value: Any) -> List[int]:
        """
        Use the index to evaluate a query.
        
        Args:
            operator: The comparison operator.
            value: The value to compare against.
            
        Returns:
            A list of row indices that satisfy the query.
        """
        if operator == "=" or operator == "==":
            # For equality, return the row indices directly
            if value in self.value_to_rows:
                return self.value_to_rows[value]
        elif operator == "LIKE":
            # For LIKE with a prefix pattern
            if isinstance(value, str) and value.endswith('%') and not value.startswith('%'):
                prefix = value[:-1]
                result = []
                for k, rows in self.value_to_rows.items():
                    if k.startswith(prefix):
                        result.extend(rows)
                return sorted(result)
        
        return []
    
    def serialize(self) -> bytes:
        """
        Serialize the index to bytes.
        
        Returns:
            The serialized index as bytes.
        """
        base_data = super().serialize()
        
        # Serialize the value-to-rows dictionary
        content = pickle.dumps(self.value_to_rows)
        content_len = struct.pack('!I', len(content))
        
        return base_data + content_len + content
    
    @classmethod
    def deserialize_content(cls, column_name: str, data: bytes) -> Tuple['DictionaryIndex', int]:
        """
        Deserialize a dictionary index from bytes.
        
        Args:
            column_name: The name of the column this index is for.
            data: The serialized index content.
            
        Returns:
            A tuple of (index, bytes consumed).
        """
        content_len = struct.unpack('!I', data[:4])[0]
        content = data[4:4+content_len]
        value_to_rows = pickle.loads(content)
        
        return cls(column_name, value_to_rows), 4 + content_len
