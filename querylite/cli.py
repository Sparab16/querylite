"""
Command-line interface for the QueryLite engine.
"""
import argparse
import sys
import os
import pandas as pd
import json
from typing import Dict, List, Any
from querylite.table import Table
from querylite.query import QueryEngine
from querylite.column import DataType, CompressionType
from querylite.index import IndexType


def load_table_from_csv(file_path: str) -> Table:
    """
    Load a table from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        A Table object loaded from the CSV file.
    """
    # Read CSV using pandas
    df = pd.read_csv(file_path)
    
    # Create a table
    table_name = os.path.splitext(os.path.basename(file_path))[0]
    table = Table(table_name)
    
    # Convert data to dictionary
    data_dict = {}
    for column in df.columns:
        data_dict[column] = df[column].tolist()
    
    # Add columns to table
    table.add_columns_from_dict(data_dict)
    
    return table


def load_table_from_json(file_path: str) -> Table:
    """
    Load a table from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        A Table object loaded from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Create a table
    table_name = os.path.splitext(os.path.basename(file_path))[0]
    table = Table(table_name)
    
    # Check if data is in records format (list of dicts) or column format (dict of lists)
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        # Convert records to column format
        columns = {}
        for key in data[0].keys():
            columns[key] = [item.get(key) for item in data]
        data = columns
    
    # Add columns to table
    if isinstance(data, dict):
        table.add_columns_from_dict(data)
    
    return table


def format_table_output(results: Dict[str, List[Any]], max_rows: int = 100, max_col_width: int = 20) -> str:
    """
    Format query results as a table.
    
    Args:
        results: Query results as a dictionary of column names to column data.
        max_rows: Maximum number of rows to display.
        max_col_width: Maximum width of each column.
        
    Returns:
        Formatted table string.
    """
    if not results:
        return "Empty result"
    
    # Get column names and determine their widths
    cols = list(results.keys())
    col_widths = {col: min(max_col_width, max(len(col), max(len(str(v)) for v in results[col][:max_rows]))) 
                 for col in cols}
    
    # Create header row
    header = " | ".join(col.ljust(col_widths[col]) for col in cols)
    separator = "-" * len(header)
    
    # Create data rows
    num_rows = min(max_rows, len(next(iter(results.values()))))
    rows = []
    for i in range(num_rows):
        row_values = [str(results[col][i])[:max_col_width].ljust(col_widths[col]) for col in cols]
        rows.append(" | ".join(row_values))
    
    # Add ellipsis row if data was truncated
    if num_rows < len(next(iter(results.values()))):
        rows.append("... ({} more rows)".format(len(next(iter(results.values()))) - num_rows))
    
    # Combine all parts
    return "\n".join([header, separator] + rows)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='QueryLite: Lightweight Columnar Storage Engine with SQL support')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert a file to the QueryLite format')
    convert_parser.add_argument('input_file', help='Input file path (CSV or JSON)')
    convert_parser.add_argument('output_file', help='Output file path for the QueryLite format')
    convert_parser.add_argument('--compress', choices=['none', 'rle', 'dict', 'auto'], default='auto',
                              help='Compression strategy (default: auto)')
    convert_parser.add_argument('--skip-compression', action='store_true',
                              help='Skip all compression for faster import of large datasets')
    convert_parser.add_argument('--description', help='Description of the table')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Execute a SQL query on a QueryLite file')
    query_parser.add_argument('file', help='QueryLite file to query')
    query_parser.add_argument('sql', help='SQL query to execute')
    query_parser.add_argument('--output', help='Output file for query results (CSV format)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show information about a QueryLite file')
    info_parser.add_argument('file', help='QueryLite file to inspect')
    info_parser.add_argument('--detailed', action='store_true', help='Show detailed column statistics')
    
    # Metadata command
    metadata_parser = subparsers.add_parser('metadata', help='View or modify metadata of a QueryLite file')
    metadata_parser.add_argument('file', help='QueryLite file to work with')
    metadata_parser.add_argument('--get', metavar='KEY', help='Get metadata value for a specific key')
    metadata_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set metadata value for a key')
    metadata_parser.add_argument('--list', action='store_true', help='List all metadata')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Create and manage indices')
    index_parser.add_argument('file', help='QueryLite file to work with')
    index_parser.add_argument('--create', nargs=2, metavar=('COLUMN', 'TYPE'), 
                             help='Create index for a column. TYPE can be minmax, bitmap, or dictionary')
    index_parser.add_argument('--auto', action='store_true', help='Automatically create appropriate indices')
    index_parser.add_argument('--list', action='store_true', help='List all indices')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark query performance')
    benchmark_parser.add_argument('file', help='QueryLite file to benchmark')
    benchmark_parser.add_argument('--query', required=True, help='SQL query to benchmark')
    benchmark_parser.add_argument('--iterations', type=int, default=5, help='Number of iterations (default: 5)')
    benchmark_parser.add_argument('--with-indices', action='store_true', help='Create indices before benchmarking')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export a QueryLite file to other formats')
    export_parser.add_argument('file', help='QueryLite file to export')
    export_parser.add_argument('output', help='Output file path')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv', help='Export format (default: csv)')
    
    args = parser.parse_args()
    
    # Process commands
    if args.command == 'convert':
        try:
            # Load source data
            # Get table name from file name
            table_name = os.path.splitext(os.path.basename(args.input_file))[0]
            
            # Create table with description if provided
            description = args.description if args.description else f"Table created from {os.path.basename(args.input_file)}"
            
            if args.input_file.lower().endswith('.csv'):
                df = pd.read_csv(args.input_file)
                
                # Create a table with description
                table = Table(table_name, description)
                
                # Convert data to dictionary
                data_dict = {}
                for column in df.columns:
                    data_dict[column] = df[column].tolist()
                
                # Add columns to table
                table.add_columns_from_dict(data_dict)
            elif args.input_file.lower().endswith('.json'):
                with open(args.input_file, 'r') as f:
                    data = json.load(f)
                
                # Create a table with description
                table = Table(table_name, description)
                
                # Check if data is in records format (list of dicts) or column format (dict of lists)
                if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    # Convert records to column format
                    columns = {}
                    for key in data[0].keys():
                        columns[key] = [item.get(key) for item in data]
                    data = columns
                
                # Add columns to table
                if isinstance(data, dict):
                    table.add_columns_from_dict(data)
            else:
                print(f"Unsupported input file format: {args.input_file}")
                return 1
            
            # Apply manual compression if specified
            # Extract column data for rebuilding the table
            data = {}
            for col_name in table.get_column_names():
                column = table.columns[col_name]
                if column.data is not None:
                    data[col_name] = column.data.tolist()
                else:
                    data[col_name] = []
            
            # Create a new table
            new_table = Table(table.name, description=description)
            
            if args.skip_compression:
                # Skip all compression for faster import
                new_table.add_columns_from_dict(data, skip_compression=True)
                print("Skipping compression for faster import")
            elif args.compress != 'auto':
                # Use the specified compression type
                compression = {
                    'none': CompressionType.NONE,
                    'rle': CompressionType.RLE,
                    'dict': CompressionType.DICT
                }[args.compress]
                
                compression_dict = {col_name: compression for col_name in data.keys()}
                new_table.add_columns_from_dict(data, compression=compression_dict)
            else:
                # Use auto compression
                new_table.add_columns_from_dict(data)
                
            table = new_table
            
            # Set source file metadata
            table.set_metadata("source_file", args.input_file)
            table.set_metadata("source_format", os.path.splitext(args.input_file)[1][1:].upper())
            
            # Update table statistics
            table.update_metadata_stats()
            
            # Save to output file
            table.save(args.output_file)
            
            print(f"Converted {args.input_file} to {args.output_file}")
            print(f"Table: {table.name}")
            print(f"Description: {table.get_metadata('description') or 'N/A'}")
            print(f"Created: {table.get_metadata('created_at')}")
            print(f"Columns: {', '.join(table.get_column_names())}")
            print(f"Rows: {len(table)}")
            
            # Show compression info
            for col_name, column in table.columns.items():
                print(f"  {col_name}: {column.dtype.name}, "
                      f"Compression: {column.compression.name}, "
                      f"Size: {column.get_size()} bytes")
            
            return 0
        
        except Exception as e:
            print(f"Error converting file: {str(e)}")
            return 1
    
    elif args.command == 'query':
        try:
            # Load table
            table = Table.load(args.file)
            
            # Set up query engine
            engine = QueryEngine()
            engine.register_table(table)
            
            # Execute query
            results = engine.execute(args.sql)
            
            # Output results
            if args.output:
                # Save to CSV
                pd.DataFrame(results).to_csv(args.output, index=False)
                print(f"Results saved to {args.output}")
            else:
                # Print to console
                print(format_table_output(results))
            
            return 0
        
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return 1
    
    elif args.command == 'info':
        try:
            # Load table
            table = Table.load(args.file)
            
            # Make sure statistics are up to date
            table.update_metadata_stats()
            
            # Display table information
            print(f"Table: {table.name}")
            print(f"Description: {table.get_metadata('description') or 'N/A'}")
            print(f"Created: {table.get_metadata('created_at')}")
            print(f"Last Modified: {table.get_metadata('modified_at')}")
            print(f"Schema Version: {table.get_metadata('schema_version')}")
            print(f"Rows: {len(table)}")
            print(f"Columns: {len(table.columns)}")
            
            # Display index information
            if table.indices:
                print("\nIndices:")
                for col_name, indices_dict in table.indices.items():
                    index_types = [idx_type.name for idx_type in indices_dict.keys()]
                    print(f"  {col_name}: {', '.join(index_types)}")
            
            # Display column information
            print("\nColumn Details:")
            
            total_size = 0
            column_stats = table.get_metadata("column_statistics") or {}
            
            for col_name, column in table.columns.items():
                col_size = column.get_size()
                total_size += col_size
                stats = column_stats.get(col_name, {})
                
                # Basic info for all columns
                print(f"  {col_name}: {column.dtype.name}, "
                      f"Compression: {column.compression.name}, "
                      f"Size: {col_size} bytes")
                
                # Show more detailed stats if requested
                if args.detailed and stats:
                    if column.dtype in (DataType.INT, DataType.FLOAT) and "min" in stats and "max" in stats:
                        print(f"    Min: {stats['min']}, Max: {stats['max']}, "
                              f"Unique Values: {stats.get('unique_count', 'N/A')}")
                    elif column.dtype == DataType.STRING and "unique_count" in stats:
                        print(f"    Unique Values: {stats['unique_count']}, "
                              f"Max Length: {stats.get('max_length', 'N/A')}, "
                              f"Avg Length: {stats.get('avg_length', 'N/A'):.1f}")
            
            print(f"\nTotal Size: {total_size} bytes ({total_size / 1024:.1f} KB)")
            
            return 0
        
        except Exception as e:
            print(f"Error reading file information: {str(e)}")
            return 1
    
    elif args.command == 'metadata':
        try:
            # Load table
            table = Table.load(args.file)
            
            if args.get:
                # Get specific metadata
                if args.get == 'custom_properties':
                    value = table.get_metadata('custom_properties')
                    print(json.dumps(value, indent=2))
                else:
                    value = table.get_metadata(args.get)
                    if value is None:
                        # Try in custom properties
                        custom_props = table.get_metadata('custom_properties') or {}
                        value = custom_props.get(args.get)
                        
                    if value is None:
                        print(f"Metadata key '{args.get}' not found")
                    else:
                        if isinstance(value, dict):
                            print(json.dumps(value, indent=2))
                        else:
                            print(value)
            
            elif args.set:
                # Set metadata
                key, value = args.set
                
                # Try to convert value to appropriate type
                try:
                    # Try as number
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                        value = float(value)
                    # Try as JSON
                    elif (value.startswith('{') and value.endswith('}')) or \
                         (value.startswith('[') and value.endswith(']')):
                        value = json.loads(value)
                except:
                    pass  # Keep as string
                
                # Set the value
                table.set_metadata(key, value)
                print(f"Set metadata {key} = {value}")
                
                # Save the table
                table.save(args.file)
                print(f"Saved changes to {args.file}")
            
            elif args.list:
                # List all metadata
                metadata = table.get_metadata()
                
                # Print built-in metadata first
                print("Built-in Metadata:")
                for key in ['created_at', 'modified_at', 'description', 
                           'schema_version', 'row_count', 'column_count']:
                    if key in metadata:
                        print(f"  {key}: {metadata[key]}")
                
                # Print custom properties
                custom_props = metadata.get('custom_properties', {})
                if custom_props:
                    print("\nCustom Properties:")
                    for key, value in custom_props.items():
                        if isinstance(value, dict):
                            print(f"  {key}: {json.dumps(value, indent=2)}")
                        else:
                            print(f"  {key}: {value}")
                
                # Print column statistics if available
                if 'column_statistics' in metadata and metadata['column_statistics']:
                    print("\nColumn Statistics: (use 'info --detailed' command to view)")
            
            else:
                print("Please specify an operation (--get, --set, or --list)")
            
            return 0
            
        except Exception as e:
            print(f"Error handling metadata: {str(e)}")
            return 1
            
    elif args.command == 'index':
        try:
            # Load table
            table = Table.load(args.file)
            
            if args.create:
                # Create a specific index
                column_name, index_type_str = args.create
                
                # Check column exists
                if column_name not in table.columns:
                    print(f"Column '{column_name}' not found in table")
                    return 1
                
                # Map string to IndexType
                index_type_map = {
                    'minmax': IndexType.MINMAX,
                    'bitmap': IndexType.BITMAP,
                    'dictionary': IndexType.DICTIONARY
                }
                
                if index_type_str.lower() not in index_type_map:
                    print(f"Unknown index type: {index_type_str}")
                    print(f"Valid types are: {', '.join(index_type_map.keys())}")
                    return 1
                    
                index_type = index_type_map[index_type_str.lower()]
                
                # Create the index
                print(f"Creating {index_type_str} index for column '{column_name}'...")
                index = table.create_index(column_name, index_type)
                
                # Save the table
                table.save(args.file)
                print(f"Index created and saved to {args.file}")
            
            elif args.auto:
                # Automatically create indices
                print("Analyzing table and creating appropriate indices...")
                created = table.create_automatic_indices()
                
                # Report created indices
                if created:
                    print("Created indices:")
                    for col_name, index_types in created.items():
                        index_names = [idx.name for idx in index_types]
                        print(f"  {col_name}: {', '.join(index_names)}")
                        
                    # Save the table
                    table.save(args.file)
                    print(f"Indices saved to {args.file}")
                else:
                    print("No suitable indices were created")
            
            elif args.list:
                # List all indices
                if not table.indices:
                    print("No indices found in table")
                else:
                    print("Indices:")
                    for col_name, indices_dict in table.indices.items():
                        for index_type, index in indices_dict.items():
                            print(f"  Column: {col_name}, Type: {index_type.name}")
                            
                            # Print index-specific details based on index type
                            # We'll just print the index type without accessing specific attributes
                            # to avoid type checker errors
                            print(f"    Index Type: {index_type.name}")
            
            else:
                print("Please specify an operation (--create, --auto, or --list)")
            
            return 0
            
        except Exception as e:
            print(f"Error handling indices: {str(e)}")
            return 1
    
    elif args.command == 'benchmark':
        try:
            import time
            
            # Load table
            table = Table.load(args.file)
            
            # Create indices if requested
            if args.with_indices:
                print("Creating automatic indices for benchmarking...")
                created = table.create_automatic_indices()
                if created:
                    for col_name, index_types in created.items():
                        index_names = [idx.name for idx in index_types]
                        print(f"  {col_name}: {', '.join(index_names)}")
            
            # Set up query engine
            engine = QueryEngine()
            engine.register_table(table)
            
            # Execute warmup query
            print("Executing warmup query...")
            engine.execute(args.query)
            
            # Benchmark
            print(f"\nBenchmarking query: {args.query}")
            print(f"Running {args.iterations} iterations...\n")
            
            times = []
            results = None  # Store the last result for display
            for i in range(args.iterations):
                start_time = time.time()
                results = engine.execute(args.query)
                end_time = time.time()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                print(f"Iteration {i+1}: {execution_time:.6f} seconds")
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Print statistics
            print("\nResults:")
            print(f"  Average execution time: {avg_time:.6f} seconds")
            print(f"  Minimum execution time: {min_time:.6f} seconds")
            print(f"  Maximum execution time: {max_time:.6f} seconds")
            print(f"  Standard deviation: {(sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5:.6f} seconds")
            
            # Print results sample
            print("\nQuery results (first few rows):")
            if results:
                print(format_table_output(results, max_rows=5))
            else:
                print("No results to display")
            
            return 0
            
        except Exception as e:
            print(f"Error benchmarking query: {str(e)}")
            return 1
    
    elif args.command == 'export':
        try:
            # Load table
            table = Table.load(args.file)
            
            # Get all data
            all_data = table.select(table.get_column_names())
            
            # Convert to DataFrame for easy export
            df = pd.DataFrame(all_data)
            
            # Export based on format
            if args.format == 'csv':
                df.to_csv(args.output, index=False)
                print(f"Exported table to CSV: {args.output}")
            elif args.format == 'json':
                # Export as column-oriented JSON (same structure as input)
                with open(args.output, 'w') as f:
                    json.dump(all_data, f, indent=2)
                print(f"Exported table to column-oriented JSON: {args.output}")
            
            return 0
            
        except Exception as e:
            print(f"Error exporting file: {str(e)}")
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
