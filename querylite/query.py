"""
Query engine module for executing SQL queries against tables.
"""
from typing import Dict, List, Any, Tuple, Optional, Callable
import re
from .parser import SQLParser, SQLQuery
from .table import Table


class QueryEngine:
    """
    Engine for executing SQL queries against tables.
    
    Attributes:
        tables: Dictionary mapping table names to Table objects.
    """
    
    def __init__(self):
        """Initialize a query engine."""
        self.tables: Dict[str, Table] = {}
        self.parser = SQLParser()
    
    def register_table(self, table: Table) -> None:
        """
        Register a table with the query engine.
        
        Args:
            table: The table to register.
        
        Raises:
            ValueError: If a table with the same name is already registered.
        """
        if table.name in self.tables:
            raise ValueError(f"Table '{table.name}' is already registered")
        
        self.tables[table.name] = table
    
    def deregister_table(self, table_name: str) -> None:
        """
        Deregister a table from the query engine.
        
        Args:
            table_name: The name of the table to deregister.
        
        Raises:
            ValueError: If no table with the given name is registered.
        """
        if table_name not in self.tables:
            raise ValueError(f"No table named '{table_name}' is registered")
        
        del self.tables[table_name]
    
    def execute(self, query: str) -> Dict[str, List[Any]]:
        """
        Execute a SQL query.
        
        Args:
            query: The SQL query to execute.
            
        Returns:
            A dictionary mapping column names to column data for the query result.
            
        Raises:
            ValueError: If the query is invalid or references a table that is not registered.
        """
        # Parse the query
        try:
            parsed_query = self.parser.parse(query)
        except Exception as e:
            raise ValueError(f"Failed to parse query: {str(e)}")
        
        # Execute the parsed query
        return self._execute_query(parsed_query)
    
    def _execute_query(self, query: SQLQuery) -> Dict[str, List[Any]]:
        """
        Execute a parsed SQL query.
        
        Args:
            query: The parsed SQL query to execute.
            
        Returns:
            A dictionary mapping column names to column data for the query result.
            
        Raises:
            ValueError: If the query references a table that is not registered.
        """
        # Check if the table exists
        if query.table_name not in self.tables:
            raise ValueError(f"Table '{query.table_name}' not found")
        
        table = self.tables[query.table_name]
        
        # Handle wildcard select
        if query.select_columns == ["*"]:
            select_columns = table.get_column_names()
        else:
            select_columns = query.select_columns
            # Check if all selected columns exist in the table
            for col in select_columns:
                if col not in table.columns:
                    raise ValueError(f"Column '{col}' not found in table '{table.name}'")
        
        # Apply where conditions
        row_indices = None
        if query.where_conditions:
            row_indices = self._apply_where_conditions(table, query.where_conditions)
        
        # Select columns and rows
        result = table.select(select_columns, row_indices)
        
        return result
    
    def _apply_where_conditions(self, table: Table, where_conditions: List[Tuple[str, str, Any]]) -> List[int]:
        """
        Apply WHERE conditions to filter rows with predicate pushdown.
        
        Args:
            table: The table to filter.
            where_conditions: List of (column_name, operator, value) tuples.
            
        Returns:
            A list of row indices that match all conditions.
        """
        # Initialize with all rows
        matching_rows = set(range(len(table)))
        
        # Apply each condition
        for col_name, op, value in where_conditions:
            # Check if column exists
            if col_name not in table.columns:
                raise ValueError(f"Column '{col_name}' not found in table '{table.name}'")
            
            # Create predicate function based on operator
            predicate = self._create_predicate(op, value)
            
            # Apply predicate to column with index-based optimization
            condition_rows = set(table.filter_rows(col_name, predicate, op, value))
            
            # Intersection with existing matching rows (AND logic)
            matching_rows &= condition_rows
        
        return sorted(matching_rows)
    
    def _create_predicate(self, op: str, value: Any) -> Callable[[Any], bool]:
        """
        Create a predicate function for a given operator and value.
        
        Args:
            op: The operator (=, !=, >, <, >=, <=, LIKE, NOT LIKE).
            value: The value to compare against.
            
        Returns:
            A function that takes a value and returns True if the condition is met.
            
        Raises:
            ValueError: If the operator is not supported.
        """
        if op == "=":
            return lambda x: x == value
        elif op in ["!=", "<>"]:
            return lambda x: x != value
        elif op == "<":
            return lambda x: x < value
        elif op == ">":
            return lambda x: x > value
        elif op == "<=":
            return lambda x: x <= value
        elif op == ">=":
            return lambda x: x >= value
        elif op.upper() == "LIKE":
            # Convert SQL LIKE pattern to regex pattern
            pattern = re.escape(value).replace("%", ".*").replace("_", ".")
            regex = re.compile(f"^{pattern}$", re.IGNORECASE)
            return lambda x: bool(regex.match(str(x)))
        elif op.upper() == "NOT LIKE":
            pattern = re.escape(value).replace("%", ".*").replace("_", ".")
            regex = re.compile(f"^{pattern}$", re.IGNORECASE)
            return lambda x: not bool(regex.match(str(x)))
        else:
            raise ValueError(f"Unsupported operator: {op}")
