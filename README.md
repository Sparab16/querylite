# QueryLite

A lightweight columnar storage engine with SQL query support, designed for writting query on CSV/JSON files.

## Features

- **Columnar Storage**: Efficient data storage in a column-oriented format
- **SQL Query Support**: Execute SQL queries on stored data
- **Multiple Data Sources**: Import data from CSV and JSON files
- **Compression Options**:
  - RLE (Run Length Encoding)
  - Dictionary-based compression
  - Automatic compression selection
- **Data Types**: Support for integers, floats, strings, and other common data types
- **Performance**: Built for efficient data processing and querying

## Installation

Install QueryLite using pip:

```bash
$ git clone https://github.com/Sparab16/querylite.git
$ cd /querylite
$ pip install .
```

Requirements:

- Python >= 3.7
- numpy
- pandas
- lark-parser

## Usage

### Converting Data Files

Convert CSV or JSON files to QueryLite format:

```bash
# Basic conversion
querylite convert input.csv output.qlite

# With specific compression
querylite convert input.csv output.qlite --compress rle

# With description
querylite convert input.csv output.qlite --description "My dataset"

# Skip compression for large datasets
querylite convert input.csv output.qlite --skip-compression
```

### Querying Data

Execute SQL queries on QueryLite files:

```bash
# Basic query
querylite query data.qlite "SELECT * FROM table"

# Save results to CSV
querylite query data.qlite "SELECT column1, column2 FROM table WHERE column1 > 100" --output results.csv
```

### Managing Indices

Create and manage indices for better query performance:

```bash
# Create specific index
querylite index data.qlite --create column_name minmax

# Create automatic indices
querylite index data.qlite --auto

# List indices
querylite index data.qlite --list
```

### File Information

View file information and statistics:

```bash
# Basic info
querylite info data.qlite

# Detailed info with column statistics
querylite info data.qlite --detailed
```

### Metadata Management

Manage file metadata:

```bash
# List metadata
querylite metadata data.qlite --list

# Get specific metadata
querylite metadata data.qlite --get key_name

# Set metadata
querylite metadata data.qlite --set key_name value
```

### Performance Benchmarking

Benchmark query performance:

```bash
# Basic benchmarking
querylite benchmark data.qlite --query "SELECT * FROM table"

# Benchmark with indices
querylite benchmark data.qlite --query "SELECT * FROM table" --with-indices

# Multiple iterations
querylite benchmark data.qlite --query "SELECT * FROM table" --iterations 10
```

### Data Export

Export data to other formats:

```bash
# Export to CSV
querylite export data.qlite output.csv

# Export to JSON
querylite export data.qlite output.json --format json
```

## SQL Support

QueryLite supports basic SQL queries including:

- SELECT statements with column selection
- FROM clause for table specification
- WHERE clause with conditions
- Comparison operators: =, !=, <>, <, >, <=, >=
- LIKE and NOT LIKE operators for pattern matching
- Basic AND conditions in WHERE clause

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

**Shreyas Parab**

- GitHub: [@Sparab16](https://github.com/Sparab16)
