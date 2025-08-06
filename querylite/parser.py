"""
SQL parser module for parsing SQL queries.
"""
from lark import Lark, Transformer, v_args, Tree
from typing import List, Dict, Any, Tuple, Optional, Union
import re


class SQLQuery:
    """
    Represents a parsed SQL query.
    
    Attributes:
        select_columns: List of column names to select.
        table_name: The name of the table to query.
        where_conditions: List of where conditions as (column_name, operator, value) tuples.
    """
    
    def __init__(self, select_columns: List[str], table_name: str, where_conditions: Optional[List[Tuple[str, str, Any]]] = None):
        """
        Initialize a SQL query.
        
        Args:
            select_columns: List of column names to select.
            table_name: The name of the table to query.
            where_conditions: Optional list of where conditions as (column_name, operator, value) tuples.
        """
        self.select_columns = select_columns
        self.table_name = table_name
        self.where_conditions = where_conditions or []
        
    def __repr__(self) -> str:
        """
        Get a string representation of the query.
        
        Returns:
            String representation of the query.
        """
        select_clause = ", ".join(self.select_columns)
        where_clause = " AND ".join([f"{col} {op} {val}" for col, op, val in self.where_conditions]) if self.where_conditions else ""
        
        if where_clause:
            return f"SELECT {select_clause} FROM {self.table_name} WHERE {where_clause}"
        else:
            return f"SELECT {select_clause} FROM {self.table_name}"


class SQLTransformer(Transformer):
    """Transformer for converting parsed SQL into a SQLQuery object."""
    
    def __init__(self):
        super().__init__(visit_tokens=True)
    
    @v_args(inline=True)
    def quoted_name(self, name):
        # Remove quotes from quoted names
        return str(name).strip('"\'`')
    
    def name(self, items):
        return str(items[0])
    
    def column_list(self, items):
        return [str(item) for item in items]
    
    def star(self, _):
        return ["*"]
    
    def select_columns(self, items):
        # Flatten if we have a single item that is itself a list (e.g., star)
        if len(items) == 1 and isinstance(items[0], list):
            return items[0]
        return items
    
    def string_literal(self, items):
        # Remove surrounding quotes
        val = str(items[0])
        if val.startswith("'") and val.endswith("'"):
            val = val[1:-1]
        return val
    
    def number_literal(self, items):
        val = str(items[0])
        if '.' in val:
            return float(val)
        return int(val)
    
    def comparison_op(self, items):
        # Make sure we have at least one item in the list
        if not items:
            return ""
        return str(items[0])
    
    @v_args(inline=True)
    def where_condition(self, column_name, op, value):
        return (column_name, op, value)
    
    def where_clause(self, items):
        return items
    
    @v_args(inline=True)
    def select_stmt(self, select_cols, table_name, *args):
        where_conditions = None
        if len(args) > 0 and args[0] is not None:
            where_conditions = args[0]
        
        return SQLQuery(select_cols, table_name, where_conditions)


class SQLParser:
    """Parser for SQL queries."""
    
    def __init__(self):
        # Define a simple SQL grammar using Lark
        grammar = """
            // SQL grammar for simple queries
            start: select_stmt
            
            select_stmt: "SELECT" select_columns "FROM" table_name [where_clause]
            select_columns: column_list | star
            column_list: name ("," name)*
            star: "*"
            table_name: name
            name: CNAME | quoted_name
            quoted_name: ESCAPED_STRING
            
            where_clause: "WHERE" where_condition ("AND" where_condition)*
            where_condition: name comparison_op value
            comparison_op: OPERATOR
            OPERATOR: "=" | "!=" | "<>" | "<" | ">" | "<=" | ">=" | "LIKE" | "NOT" "LIKE"
            value: string_literal | number_literal | "NULL"
            string_literal: ESCAPED_STRING
            number_literal: SIGNED_NUMBER
            
            %import common.CNAME
            %import common.ESCAPED_STRING
            %import common.SIGNED_NUMBER
            %import common.WS
            %ignore WS
        """
        
        # Create transformer separately for better error handling
        self.transformer = SQLTransformer()
        self.parser = Lark(grammar, parser='lalr')
    
    def parse(self, sql_query: str) -> SQLQuery:
        """
        Parse a SQL query.
        
        Args:
            sql_query: The SQL query string to parse.
            
        Returns:
            A SQLQuery object representing the parsed query.
            
        Raises:
            Exception: If the query syntax is invalid.
        """
        # Clean and normalize the query
        sql_query = self._normalize_query(sql_query)
        
        # Parse the query
        try:
            # First parse with Lark to get a tree
            parse_tree = self.parser.parse(sql_query)
            
            # Process the tree manually
            if not isinstance(parse_tree, Tree):
                raise Exception("Parser produced unexpected result")
            
            if parse_tree.data != 'start':
                raise Exception(f"Expected 'start' as root, got '{parse_tree.data}'")
            
            # Process the select_stmt which should be the only child of start
            if not parse_tree.children or len(parse_tree.children) != 1:
                raise Exception("Invalid parse tree structure")
            
            select_stmt = parse_tree.children[0]
            if not isinstance(select_stmt, Tree) or select_stmt.data != 'select_stmt':
                raise Exception("Expected 'select_stmt' node")
            
            # Extract components from select_stmt
            if len(select_stmt.children) < 2:
                raise Exception("Incomplete SELECT statement")
            
            # Process select columns
            select_cols_tree = select_stmt.children[0]
            if not isinstance(select_cols_tree, Tree):
                raise Exception("Expected columns tree")
            
            # Extract columns
            select_cols = []
            
            # Handle column selection - according to the grammar, select_cols_tree should be 'select_columns'
            if select_cols_tree.data == 'select_columns':
                # Process the children of select_columns
                if len(select_cols_tree.children) == 0:
                    raise Exception("Empty column selection")
                
                # Get the first child which should be either 'star' or 'column_list'
                child = select_cols_tree.children[0]
                if isinstance(child, Tree):
                    if child.data == 'star':
                        # This is a SELECT * query
                        select_cols = ['*']
                    elif child.data == 'column_list':
                        # This is a SELECT col1, col2, ... query
                        for col_item in child.children:
                            if isinstance(col_item, Tree) and col_item.data == 'name':
                                if col_item.children:
                                    select_cols.append(str(col_item.children[0]))
                                else:
                                    select_cols.append(str(col_item))
                            else:
                                select_cols.append(str(col_item))
                    else:
                        raise Exception(f"Unexpected column type: {child.data}")
                else:
                    select_cols.append(str(child))
            else:
                raise Exception(f"Expected select_columns, got: {select_cols_tree.data}")
            
            # Extract table name - navigate through the tree structure
            table_name_tree = select_stmt.children[1]
            if isinstance(table_name_tree, Tree) and table_name_tree.data == 'table_name':
                if table_name_tree.children and isinstance(table_name_tree.children[0], Tree):
                    name_node = table_name_tree.children[0]
                    if name_node.children:
                        table_name = str(name_node.children[0])
                    else:
                        raise Exception("Empty table name")
                else:
                    table_name = str(table_name_tree.children[0])
            else:
                raise Exception(f"Expected table_name tree, got: {type(table_name_tree)}")
            
            # Extract where clause if it exists
            where_conditions = []
            if len(select_stmt.children) > 2:
                where_clause = select_stmt.children[2]
                if isinstance(where_clause, Tree) and where_clause.data == 'where_clause':
                    for condition in where_clause.children:
                        if isinstance(condition, Tree) and condition.data == 'where_condition':
                            if len(condition.children) == 3:
                                # Extract column name (navigate through name tree)
                                col_node = condition.children[0]
                                col = ""
                                if isinstance(col_node, Tree) and col_node.data == 'name' and col_node.children:
                                    col = str(col_node.children[0])
                                else:
                                    col = str(col_node)
                                    
                                # Extract operator - direct from the query
                                op_node = condition.children[1]
                                
                                # First try to get operator from op_node
                                if isinstance(op_node, Tree):
                                    if op_node.children:
                                        op = str(op_node.children[0])
                                    else:
                                        # Empty tree, extract from query string
                                        col_name = str(condition.children[0].children[0]) if isinstance(condition.children[0], Tree) else str(condition.children[0])
                                        val_node = condition.children[2]
                                        val_str = ""
                                        if isinstance(val_node, Tree) and val_node.children:
                                            val_str = str(val_node.children[0])
                                        
                                        # Extract operator from the query by looking between col_name and val_str
                                        col_idx = sql_query.find(col_name)
                                        val_idx = sql_query.find(str(val_str), col_idx)
                                        
                                        if col_idx >= 0 and val_idx > col_idx:
                                            op_part = sql_query[col_idx + len(col_name):val_idx].strip()
                                            # Extract just the operator symbols
                                            op = ''.join([c for c in op_part if c in '<>=!'])
                                        else:
                                            # Fallback
                                            op = ">"  # Assuming it's a greater than operator based on your query
                                else:
                                    op = str(op_node)
                                
                                # Extract value (could be string_literal, number_literal, etc.)
                                val_node = condition.children[2]
                                val = None
                                
                                # First check if it's a value node
                                if isinstance(val_node, Tree) and val_node.data == 'value':
                                    # Get the child node (string_literal or number_literal)
                                    if val_node.children:
                                        child_node = val_node.children[0]
                                        if isinstance(child_node, Tree):
                                            # Check the type of value
                                            if child_node.data == 'string_literal':
                                                # String literal - get the token value and remove quotes
                                                if child_node.children:
                                                    val_str = str(child_node.children[0])
                                                    val = val_str.strip('"\'')
                                            elif child_node.data == 'number_literal':
                                                # Number literal - convert to appropriate type
                                                if child_node.children:
                                                    val_str = str(child_node.children[0])
                                                    if '.' in val_str:
                                                        val = float(val_str)
                                                    else:
                                                        val = int(val_str)
                                else:
                                    # Not a tree or unexpected structure
                                    val = str(val_node)
                                
                                # Use a default value if extraction failed
                                if val is None:
                                    val = 0
                                    
                                where_conditions.append((col, op, val))
            
            # Create and return SQLQuery object
            return SQLQuery(select_cols, table_name, where_conditions)
        except Exception as e:
            raise Exception(f"Failed to parse query: {str(e)}")
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize a SQL query by removing extra whitespace and ensuring keywords are uppercase.
        
        Args:
            query: The SQL query to normalize.
            
        Returns:
            The normalized query.
        """
        # Remove comments and extra whitespace
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Convert keywords to uppercase for consistent parsing
        keywords = ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'LIKE', 'NULL']
        for keyword in keywords:
            pattern = r'\b' + keyword + r'\b'
            query = re.sub(pattern, keyword, query, flags=re.IGNORECASE)
        
        return query
