"""
Table module for handling table data storage and operations.
"""
import os
import numpy as np
from typing import List, Dict, Union, Tuple, Any, Optional, Set, Type
import struct
import json
import datetime
from .column import Column, DataType, CompressionType
from .index import Index, IndexType, MinMaxIndex, BitmapIndex, DictionaryIndex
from .serialize import Serializer
from .deserialize import Deserializer


class Table:
    """
    Represents a table in a columnar storage format.
    
    Attributes:
        name: The name of the table.
        columns: Dictionary of column objects with column names as keys.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a table with a name.
        
        Args:
            name: The name of the table.
            description: Optional description of the table's purpose or contents.
        """
        self.name = name
        self.columns: Dict[str, Column] = {}
        self.indices: Dict[str, Dict[IndexType, Index]] = {}  # Maps column_name to dict of indices by type
        self.num_rows = 0
        
        # Initialize metadata with basic information
        timestamp = datetime.datetime.now().isoformat()
        self.metadata = {
            "created_at": timestamp,
            "modified_at": timestamp,
            "description": description,
            "schema_version": "1.0.0",
            "row_count": 0,
            "column_count": 0,
            "custom_properties": {}
        }
    
    def add_column(self, column: Column) -> None:
        """
        Add a column to the table.
        
        Args:
            column: The column to add.
        
        Raises:
            ValueError: If a column with the same name already exists.
        """
        if column.name in self.columns:
            raise ValueError(f"Column '{column.name}' already exists in table '{self.name}'")
        
        self.columns[column.name] = column
        
        # Ensure all columns have the same number of rows
        if column.data is not None:
            if self.num_rows == 0:
                self.num_rows = len(column.data)
            elif len(column.data) != self.num_rows:
                raise ValueError(f"Column '{column.name}' has {len(column.data)} rows, "
                                 f"but table '{self.name}' has {self.num_rows} rows")
        
        # Update metadata
        self.metadata["column_count"] = len(self.columns)
        self.metadata["row_count"] = self.num_rows
        self._update_modified_timestamp()
    
    def add_columns_from_dict(self, data: Dict[str, List], dtypes: Optional[Dict[str, DataType]] = None,
                              compression: Optional[Dict[str, CompressionType]] = None, 
                              skip_compression: bool = False,
                              compression_threshold: int = 1000000) -> None:
        """
        Add multiple columns to the table from a dictionary.
        
        Args:
            data: Dictionary mapping column names to column data.
            dtypes: Optional dictionary mapping column names to data types.
            compression: Optional dictionary mapping column names to compression types.
            skip_compression: If True, all columns will be stored uncompressed for faster import.
            compression_threshold: Maximum number of rows for automatic compression. Columns with
                                  more rows than this threshold will default to no compression.
        """
        # Detect the number of rows
        if not data:
            return
        
        sample_col = next(iter(data.values()))
        self.num_rows = len(sample_col)
        
        # Check all columns have the same length
        for col_name, col_data in data.items():
            if len(col_data) != self.num_rows:
                raise ValueError(f"Column '{col_name}' has {len(col_data)} rows, but expected {self.num_rows}")
        
        # Add each column
        for col_name, col_data in data.items():
            # Auto-detect data type if not provided
            if dtypes and col_name in dtypes:
                dtype = dtypes[col_name]
            else:
                dtype = self._detect_dtype(col_data)
            
            # Get compression type
            if skip_compression or len(col_data) > compression_threshold:
                comp = CompressionType.NONE
            elif compression and col_name in compression:
                comp = compression[col_name]
            else:
                # Auto-select compression based on data type
                comp = self._select_compression(dtype, col_data)
            
            # Create and add the column
            column = Column(col_name, dtype, col_data, comp)
            self.columns[col_name] = column
        
        # Update metadata
        self.metadata["column_count"] = len(self.columns)
        self.metadata["row_count"] = self.num_rows
        self._update_modified_timestamp()
    
    def _detect_dtype(self, data: List) -> DataType:
        """
        Detect the data type of a column based on its data.
        
        Args:
            data: The column data.
            
        Returns:
            The detected data type.
        """
        if not data:
            return DataType.STRING
        
        sample = data[0]
        if isinstance(sample, int):
            return DataType.INT
        elif isinstance(sample, float):
            return DataType.FLOAT
        else:
            return DataType.STRING
    
    def _select_compression(self, dtype: DataType, data: List) -> CompressionType:
        """
        Select an appropriate compression algorithm based on data type and content.
        Uses sampling for large datasets to improve performance.
        
        Args:
            dtype: The data type of the column.
            data: The column data.
            
        Returns:
            The selected compression type.
        """
        if not data:
            return CompressionType.NONE
        
        # Skip compression for very large columns (more than 1M elements)
        # This prevents excessive memory usage and processing time during import
        if len(data) > 1000000:
            return CompressionType.NONE
        
        # For medium-sized columns (more than 100K elements), use sampling
        # to determine compression type without scanning the entire column
        sample_size = 10000
        use_sampling = len(data) > 100000
        
        # For string data, dictionary encoding is usually better
        if dtype == DataType.STRING:
            if use_sampling:
                # Take a sample of the data for analysis
                import random
                sample = random.sample(data, min(sample_size, len(data)))
                unique_ratio = len(set(sample)) / len(sample)
            else:
                unique_ratio = len(set(data)) / len(data)
                
            if unique_ratio < 0.5:  # If less than 50% of values are unique
                return CompressionType.DICT
            
        elif dtype == DataType.INT or dtype == DataType.FLOAT:
            # For numeric data, check for repeated values (RLE works well with repeated values)
            if use_sampling:
                # Sample the data at regular intervals instead of checking each element
                sample_interval = max(1, len(data) // sample_size)
                sampled_data = [data[i] for i in range(0, len(data), sample_interval)]
                
                repeated_values = 0
                for i in range(1, len(sampled_data)):
                    if sampled_data[i] == sampled_data[i-1]:
                        repeated_values += 1
                
                if repeated_values > len(sampled_data) / 4:  # If more than 25% of values are repeated
                    return CompressionType.RLE
            else:
                # For smaller datasets, do a complete scan
                repeated_values = 0
                for i in range(1, len(data)):
                    if data[i] == data[i-1]:
                        repeated_values += 1
            
                if repeated_values > len(data) / 4:  # If more than 25% of values are repeated
                    return CompressionType.RLE
        
        return CompressionType.NONE  # Default to no compression for other types
    
    def get_column(self, name: str) -> Optional[Column]:
        """
        Get a column by name.
        
        Args:
            name: The name of the column.
            
        Returns:
            The column, or None if not found.
        """
        return self.columns.get(name)
    
    def get_column_names(self) -> List[str]:
        """
        Get the names of all columns in the table.
        
        Returns:
            List of column names.
        """
        return list(self.columns.keys())
    
    def save(self, file_path: str) -> None:
        """
        Save the table to a file.
        
        Args:
            file_path: The path to save the table to.
        """
        # Update metadata statistics before saving
        self.update_metadata_stats()
        
        with open(file_path, 'wb') as f:
            # Write table header
            table_name_bytes = self.name.encode('utf-8')
            header = struct.pack(f'!I{len(table_name_bytes)}sI',len(table_name_bytes), table_name_bytes, self.num_rows)
            f.write(header)
            
            # Write metadata
            metadata_bytes = json.dumps(self.metadata).encode('utf-8')
            f.write(struct.pack(f'!I{len(metadata_bytes)}s', len(metadata_bytes), metadata_bytes))
            
            # Write number of columns
            f.write(struct.pack('!I', len(self.columns)))
            
            # Write each column
            for col_name, column in self.columns.items():
                col_bytes = Serializer.serialize(column)
                f.write(col_bytes)
            
            # Write number of columns with indices
            # f.write(struct.pack('!I', len(self.indices)))
            
            # # Write each column's indices
            # for col_name, indices_dict in self.indices.items():
            #     # Write column name
            #     col_name_bytes = col_name.encode('utf-8')
            #     f.write(struct.pack(f'!I{len(col_name_bytes)}s', len(col_name_bytes), col_name_bytes))
                
            #     # Write number of indices for this column
            #     f.write(struct.pack('!I', len(indices_dict)))
                
            #     # Write each index
            #     for index_type, index in indices_dict.items():
            #         index_bytes = index.serialize()
            #         f.write(index_bytes)
    
    @classmethod
    def load(cls, file_path: str) -> 'Table':
        """
        Load a table from a file.
        
        Args:
            file_path: The path to load the table from.
            
        Returns:
            The loaded table.
        """
        with open(file_path, 'rb') as f:
            # Read table header
            name_length = struct.unpack('!I', f.read(4))[0]
            name = f.read(name_length).decode('utf-8')
            num_rows = struct.unpack('!I', f.read(4))[0]
            
            # Create table
            table = cls(name)
            table.num_rows = num_rows
            
            # Read metadata
            metadata_length = struct.unpack('!I', f.read(4))[0]
            metadata_bytes = f.read(metadata_length)
            table.metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Read columns
            num_columns = struct.unpack('!I', f.read(4))[0]
            
            for _ in range(num_columns):
                # Read column data
                data = f.read()  # Read the rest of the file
                column, bytes_consumed = Deserializer.deserialize(data)
                table.columns[column.name] = column
                
                # Move the file pointer back
                f.seek(-(len(data) - bytes_consumed), os.SEEK_CUR)
            
            try:
                # Read number of columns with indices (might not exist in older files)
                num_indexed_columns = struct.unpack('!I', f.read(4))[0]
                
                # Read each column's indices
                for _ in range(num_indexed_columns):
                    # Read column name
                    col_name_length = struct.unpack('!I', f.read(4))[0]
                    col_name = f.read(col_name_length).decode('utf-8')
                    
                    # Create entry for this column
                    if col_name not in table.indices:
                        table.indices[col_name] = {}
                    
                    # Read number of indices for this column
                    num_indices = struct.unpack('!I', f.read(4))[0]
                    
                    # Read each index
                    for _ in range(num_indices):
                        # Read index data
                        data = f.read()  # Read the rest of the file
                        index, bytes_consumed = Index.deserialize(data)
                        table.indices[col_name][index.index_type] = index
                        
                        # Move the file pointer back
                        f.seek(-(len(data) - bytes_consumed), os.SEEK_CUR)
            except:
                # If we can't read indices, it might be an older file format
                # Just continue without indices
                pass
        
        return table
    
    def create_index(self, column_name: str, index_type: IndexType) -> Index:
        """
        Create an index for a column.
        
        Args:
            column_name: The name of the column to create an index for.
            index_type: The type of index to create.
            
        Returns:
            The created index.
            
        Raises:
            ValueError: If the column doesn't exist.
        """
        if column_name not in self.columns:
            raise ValueError(f"Column '{column_name}' not found in table '{self.name}'")
            
        column = self.columns[column_name]
        if column.data is None:
            raise ValueError(f"Column '{column_name}' has no data")
            
        # Initialize column entry in indices if it doesn't exist
        if column_name not in self.indices:
            self.indices[column_name] = {}
            
        # Create the appropriate type of index
        index = None
        if index_type == IndexType.MINMAX:
            if not len(column.data):
                raise ValueError(f"Column '{column_name}' is empty")
            min_value = min(column.data)
            max_value = max(column.data)
            index = MinMaxIndex(column_name, min_value, max_value)
            
        elif index_type == IndexType.BITMAP:
            value_to_bitmap = {}
            unique_values = set(column.data)
            for value in unique_values:
                bitmap = np.zeros(len(column.data), dtype=bool)
                for i, col_value in enumerate(column.data):
                    if col_value == value:
                        bitmap[i] = True
                value_to_bitmap[value] = bitmap
            index = BitmapIndex(column_name, value_to_bitmap)
            
        elif index_type == IndexType.DICTIONARY:
            value_to_rows = {}
            for i, value in enumerate(column.data):
                if value not in value_to_rows:
                    value_to_rows[value] = []
                value_to_rows[value].append(i)
            index = DictionaryIndex(column_name, value_to_rows)
            
        if index is not None:
            self.indices[column_name][index_type] = index
            return index
            
        raise ValueError(f"Unsupported index type: {index_type}")
    
    def get_index(self, column_name: str, index_type: IndexType) -> Optional[Index]:
        """
        Get an index for a column if it exists.
        
        Args:
            column_name: The name of the column.
            index_type: The type of index to get.
            
        Returns:
            The index, or None if it doesn't exist.
        """
        if column_name not in self.indices:
            return None
            
        return self.indices[column_name].get(index_type)
    
    def filter_rows(self, column_name: str, predicate_fn, operator: Optional[str] = None, value: Any = None) -> List[int]:
        """
        Filter rows based on a predicate function applied to a column.
        Uses indices if available and applicable.
        
        Args:
            column_name: The name of the column to filter on.
            predicate_fn: A function that takes a value and returns a boolean.
            operator: Optional operator string for index-based filtering.
            value: Optional value for index-based filtering.
            
        Returns:
            A list of row indices that match the predicate.
        """
        if column_name not in self.columns:
            raise ValueError(f"Column '{column_name}' not found in table '{self.name}'")
        
        column = self.columns[column_name]
        if column.data is None:
            return []
        
        # Try to use indices if operator and value are provided
        if operator is not None and value is not None and column_name in self.indices:
            # Check each available index to see if it can satisfy the query
            for index_type, index in self.indices[column_name].items():
                if index.can_satisfy(operator, value):
                    result = index.evaluate(operator, value)
                    if result:  # If the index provided results, use them
                        return result
        
        # Fall back to full scan if no index is available or applicable
        return [i for i, value in enumerate(column.data) if predicate_fn(value)]
    
    def select(self, column_names: List[str], row_indices: Optional[List[int]] = None) -> Dict[str, List]:
        """
        Select columns and filter rows from the table.
        
        Args:
            column_names: The names of the columns to select.
            row_indices: Optional list of row indices to select. If None, all rows are selected.
            
        Returns:
            A dictionary mapping column names to column data for the selected rows.
        """
        result = {}
        
        for name in column_names:
            if name not in self.columns:
                raise ValueError(f"Column '{name}' not found in table '{self.name}'")
            
            column = self.columns[name]
            if column.data is None:
                result[name] = []
                continue
                
            if row_indices is None:
                # Select all rows
                result[name] = column.data.tolist()
            else:
                # Select specified rows
                result[name] = [column.data[i] for i in row_indices if i < len(column.data)]
        
        return result
    
    def __len__(self) -> int:
        """
        Get the number of rows in the table.
        
        Returns:
            The number of rows in the table.
        """
        return self.num_rows
    
    def get_size(self) -> int:
        """
        Get the total memory size of the table.
        
        Returns:
            The total memory size in bytes.
        """
        return sum(col.get_size() for col in self.columns.values())
    
    def get_metadata(self, key: Optional[str] = None) -> Any:
        """
        Get table metadata.
        
        Args:
            key: Optional key to retrieve specific metadata value.
                 If None, returns the entire metadata dictionary.
                 
        Returns:
            The requested metadata value or the entire metadata dictionary.
        """
        if key is None:
            return self.metadata
        
        if key == "custom_properties":
            return self.metadata.get("custom_properties", {})
            
        return self.metadata.get(key)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata value.
        
        Args:
            key: The metadata key to set.
            value: The value to set.
        """
        # If key is a built-in metadata field, update it directly
        if key in self.metadata and key != "custom_properties":
            self.metadata[key] = value
            self._update_modified_timestamp()
            return
            
        # Otherwise, store it in custom_properties
        if "custom_properties" not in self.metadata:
            self.metadata["custom_properties"] = {}
            
        self.metadata["custom_properties"][key] = value
        self._update_modified_timestamp()
    
    def update_metadata_stats(self) -> None:
        """
        Update table statistics in metadata.
        """
        self.metadata["row_count"] = self.num_rows
        self.metadata["column_count"] = len(self.columns)
        
        # Calculate column-level statistics
        column_stats = {}
        for name, column in self.columns.items():
            if column.data is not None and len(column.data) > 0:
                try:
                    # Basic statistics
                    col_stats = {
                        "type": column.dtype.name,
                        "compression": column.compression.name if column.compression else "NONE",
                        "size_bytes": column.get_size(),
                        "null_count": 0  # Not tracking nulls yet but prepared for future
                    }
                    
                    # Add specific statistics based on data type
                    if column.dtype in (DataType.INT, DataType.FLOAT):
                        col_stats.update({
                            "min": float(min(column.data)),
                            "max": float(max(column.data)),
                            "unique_count": len(set(column.data))
                        })
                    elif column.dtype == DataType.STRING:
                        col_stats.update({
                            "unique_count": len(set(column.data)),
                            "max_length": max(len(str(x)) for x in column.data),
                            "avg_length": sum(len(str(x)) for x in column.data) / len(column.data)
                        })
                    
                    column_stats[name] = col_stats
                except Exception:
                    # Skip statistics calculation if it fails
                    pass
        
        self.metadata["column_statistics"] = column_stats
        self._update_modified_timestamp()
    
    def _update_modified_timestamp(self) -> None:
        """
        Update the modified_at timestamp in metadata.
        """
        self.metadata["modified_at"] = datetime.datetime.now().isoformat()
        
    def create_automatic_indices(self) -> Dict[str, List[IndexType]]:
        """
        Analyze columns and create appropriate indices based on data characteristics.
        
        Returns:
            A dictionary mapping column names to list of created index types.
        """
        created_indices = {}
        
        for col_name, column in self.columns.items():
            if column.data is None or len(column.data) == 0:
                continue
                
            created_for_column = []
            
            # Create MinMaxIndex for all numeric columns (for range queries)
            if column.dtype in (DataType.INT, DataType.FLOAT):
                try:
                    self.create_index(col_name, IndexType.MINMAX)
                    created_for_column.append(IndexType.MINMAX)
                except Exception:
                    pass
                    
            # Count unique values to determine if bitmap index is appropriate
            unique_values = set(column.data)
            unique_ratio = len(unique_values) / len(column.data)
            
            # If number of unique values is small relative to dataset size,
            # create a bitmap index (good for low cardinality columns)
            if unique_ratio < 0.01:  # Less than 1% unique values
                try:
                    self.create_index(col_name, IndexType.BITMAP)
                    created_for_column.append(IndexType.BITMAP)
                except Exception:
                    pass
                    
            # For string columns with moderate cardinality, dictionary index works well
            if column.dtype == DataType.STRING and unique_ratio < 0.5:
                try:
                    self.create_index(col_name, IndexType.DICTIONARY)
                    created_for_column.append(IndexType.DICTIONARY)
                except Exception:
                    pass
                    
            if created_for_column:
                created_indices[col_name] = created_for_column
                
        return created_indices
