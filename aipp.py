#!/usr/bin/env python3
"""
AI++ Programming Language Interpreter - Jack Lockwood 2025
A language optimized for AI-to-AI communication with:
- Loops (while/for)
- File I/O operations
- Network/API calls
- Multi-threaded execution
- GPU acceleration support
"""

import re
import math
import random
import json
import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Optional imports for enhanced features
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' not installed. Network features disabled.")

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = np  # Fallback to numpy
    print("Warning: 'cupy' not installed. Using CPU (NumPy) instead.")


# ============================================================================
# LEXER - Enhanced Tokenizer
# ============================================================================

@dataclass
class Token:
    type: str
    value: Any
    line: int
    column: int


class Lexer:
    """Converts AI++ source code into tokens."""
    
    TOKEN_PATTERNS = [
        # Semantic vectors: âŸ¨valuesâŸ© or <values>
        (r'âŸ¨([^âŸ©]+)âŸ©', 'VECTOR'),
        (r'<([^>]+)>', 'VECTOR'),
        # Probabilistic operators
        (r'~', 'PROBABILISTIC'),
        (r'Î¸|theta', 'THRESHOLD'),
        (r'âˆ‚|partial', 'PARTIAL'),
        (r'âŠ—|tensor', 'TENSOR'),
        (r'âŠ•|combine', 'COMBINE'),
        (r'â†’|arrow', 'ARROW'),
        (r'âŠ¥|orthogonal', 'ORTHOGONAL'),
        # Keywords (ENHANCED: added for, break, continue, async, await, parallel)
        (r'\b(execute|Execute|if|else|while|for|in|break|continue|function|return|async|await|parallel)\b', 'KEYWORD'),
        # Identifiers
        (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER'),
        # Numbers (including floats)
        (r'-?\d+\.?\d*', 'NUMBER'),
        # Strings
        (r'"([^"]*)"', 'STRING'),
        (r"'([^']*)'", 'STRING'),
        # Operators
        (r'\+|\-|\*|/|%|==|!=|<=|>=|<|>|=|\|', 'OPERATOR'),
        # Delimiters
        (r'\(', 'LPAREN'),
        (r'\)', 'RPAREN'),
        (r'\{', 'LBRACE'),
        (r'\}', 'RBRACE'),
        (r'\[', 'LBRACKET'),
        (r'\]', 'RBRACKET'),
        (r',', 'COMMA'),
        (r';', 'SEMICOLON'),
        (r':', 'COLON'),
        # Whitespace (ignored)
        (r'\s+', 'WHITESPACE'),
        # Comments
        (r'#[^\n]*', 'COMMENT'),
    ]
    
    def __init__(self, source: str):
        self.source = source
        self.tokens = []
        self.position = 0
        self.line = 1
        self.column = 1
        
    def tokenize(self) -> List[Token]:
        """Convert source code into tokens."""
        while self.position < len(self.source):
            matched = False
            
            for pattern, token_type in self.TOKEN_PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(self.source, self.position)
                
                if match:
                    value = match.group(0)
                    
                    # Skip whitespace and comments
                    if token_type not in ('WHITESPACE', 'COMMENT'):
                        # Extract vector values if it's a VECTOR token
                        if token_type == 'VECTOR':
                            vector_content = match.group(1)
                            value = [float(x.strip()) for x in vector_content.split(',')]
                        # Extract string content
                        elif token_type == 'STRING':
                            value = match.group(1)
                        
                        self.tokens.append(Token(token_type, value, self.line, self.column))
                    
                    # Update position
                    self.position = match.end()
                    
                    # Track line/column for error messages
                    if '\n' in match.group(0):
                        self.line += match.group(0).count('\n')
                        self.column = 1
                    else:
                        self.column += len(match.group(0))
                    
                    matched = True
                    break
            
            if not matched:
                raise SyntaxError(f"Unexpected character '{self.source[self.position]}' at line {self.line}, column {self.column}")
        
        return self.tokens


# ============================================================================
# PARSER - Enhanced AST Builder
# ============================================================================

@dataclass
class ASTNode:
    """Base class for AST nodes."""
    type: str
    
@dataclass
class VectorNode(ASTNode):
    values: List[float]
    
@dataclass
class NumberNode(ASTNode):
    value: float
    
@dataclass
class StringNode(ASTNode):
    value: str
    
@dataclass
class IdentifierNode(ASTNode):
    name: str

@dataclass
class ListNode(ASTNode):
    elements: List[ASTNode]
    
@dataclass
class BinaryOpNode(ASTNode):
    operator: str
    left: ASTNode
    right: ASTNode
    
@dataclass
class ProbabilisticNode(ASTNode):
    expression: ASTNode
    distribution: str = "normal"
    
@dataclass
class ExecuteNode(ASTNode):
    function: ASTNode
    args: List[ASTNode]
    constraints: Optional[Dict] = None
    
@dataclass
class FunctionDefNode(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]
    is_async: bool = False
    
@dataclass
class IfNode(ASTNode):
    condition: ASTNode
    then_branch: List[ASTNode]
    else_branch: Optional[List[ASTNode]] = None

@dataclass
class WhileNode(ASTNode):
    condition: ASTNode
    body: List[ASTNode]

@dataclass
class ForNode(ASTNode):
    variable: str
    iterable: ASTNode
    body: List[ASTNode]

@dataclass
class BreakNode(ASTNode):
    pass

@dataclass
class ContinueNode(ASTNode):
    pass

@dataclass
class ReturnNode(ASTNode):
    value: Optional[ASTNode] = None
    
@dataclass
class AssignmentNode(ASTNode):
    name: str
    value: ASTNode

@dataclass
class ParallelNode(ASTNode):
    tasks: List[ASTNode]


class Parser:
    """Parses tokens into an Abstract Syntax Tree."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        
    def current_token(self) -> Optional[Token]:
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None
    
    def advance(self):
        self.position += 1
        
    def expect(self, token_type: str) -> Token:
        token = self.current_token()
        if not token or token.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token.type if token else 'EOF'}")
        self.advance()
        return token
    
    def parse(self) -> List[ASTNode]:
        """Parse all statements."""
        statements = []
        while self.current_token():
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        return statements
    
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement."""
        token = self.current_token()
        if not token:
            return None
        
        # Function definition
        if token.type == 'KEYWORD' and token.value == 'function':
            return self.parse_function_def()
        
        # Async function definition
        if token.type == 'KEYWORD' and token.value == 'async':
            self.advance()
            return self.parse_function_def(is_async=True)
        
        # If statement
        if token.type == 'KEYWORD' and token.value == 'if':
            return self.parse_if()
        
        # While loop
        if token.type == 'KEYWORD' and token.value == 'while':
            return self.parse_while()
        
        # For loop
        if token.type == 'KEYWORD' and token.value == 'for':
            return self.parse_for()
        
        # Break statement
        if token.type == 'KEYWORD' and token.value == 'break':
            self.advance()
            if self.current_token() and self.current_token().type == 'SEMICOLON':
                self.advance()
            return BreakNode('break')
        
        # Continue statement
        if token.type == 'KEYWORD' and token.value == 'continue':
            self.advance()
            if self.current_token() and self.current_token().type == 'SEMICOLON':
                self.advance()
            return ContinueNode('continue')
        
        # Return statement
        if token.type == 'KEYWORD' and token.value == 'return':
            self.advance()
            value = None
            if self.current_token() and self.current_token().type != 'SEMICOLON':
                value = self.parse_expression()
            if self.current_token() and self.current_token().type == 'SEMICOLON':
                self.advance()
            return ReturnNode('return', value)
        
        # Parallel execution
        if token.type == 'KEYWORD' and token.value == 'parallel':
            return self.parse_parallel()
        
        # Execute statement
        if token.type == 'KEYWORD' and token.value.lower() == 'execute':
            return self.parse_execute()
        
        # Assignment
        if token.type == 'IDENTIFIER':
            # Look ahead for assignment operator
            if self.position + 1 < len(self.tokens):
                next_token = self.tokens[self.position + 1]
                if next_token.type == 'OPERATOR' and next_token.value == '=':
                    return self.parse_assignment()
        
        # Expression statement
        if token.type == 'KEYWORD':
            raise SyntaxError(f"Unexpected keyword: {token.value}")
        
        expr = self.parse_expression()
        if self.current_token() and self.current_token().type == 'SEMICOLON':
            self.advance()
        return expr
    
    def parse_function_def(self, is_async: bool = False) -> FunctionDefNode:
        """Parse function definition."""
        self.expect('KEYWORD')  # 'function'
        name = self.expect('IDENTIFIER').value
        
        self.expect('LPAREN')
        params = []
        while self.current_token() and self.current_token().type != 'RPAREN':
            params.append(self.expect('IDENTIFIER').value)
            if self.current_token() and self.current_token().type == 'COMMA':
                self.advance()
        self.expect('RPAREN')
        
        self.expect('LBRACE')
        body = []
        while self.current_token() and self.current_token().type != 'RBRACE':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.expect('RBRACE')
        
        return FunctionDefNode('function_def', name, params, body, is_async)
    
    def parse_if(self) -> IfNode:
        """Parse if statement."""
        self.expect('KEYWORD')  # 'if'
        self.expect('LPAREN')
        condition = self.parse_expression()
        self.expect('RPAREN')
        
        self.expect('LBRACE')
        then_branch = []
        while self.current_token() and self.current_token().type != 'RBRACE':
            stmt = self.parse_statement()
            if stmt:
                then_branch.append(stmt)
        self.expect('RBRACE')
        
        else_branch = None
        if self.current_token() and self.current_token().type == 'KEYWORD' and self.current_token().value == 'else':
            self.advance()
            self.expect('LBRACE')
            else_branch = []
            while self.current_token() and self.current_token().type != 'RBRACE':
                stmt = self.parse_statement()
                if stmt:
                    else_branch.append(stmt)
            self.expect('RBRACE')
        
        return IfNode('if', condition, then_branch, else_branch)
    
    def parse_while(self) -> WhileNode:
        """Parse while loop."""
        self.expect('KEYWORD')  # 'while'
        self.expect('LPAREN')
        condition = self.parse_expression()
        self.expect('RPAREN')
        
        self.expect('LBRACE')
        body = []
        while self.current_token() and self.current_token().type != 'RBRACE':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.expect('RBRACE')
        
        return WhileNode('while', condition, body)
    
    def parse_for(self) -> ForNode:
        """Parse for loop."""
        self.expect('KEYWORD')  # 'for'
        self.expect('LPAREN')
        variable = self.expect('IDENTIFIER').value
        self.expect('KEYWORD')  # 'in'
        iterable = self.parse_expression()
        self.expect('RPAREN')
        
        self.expect('LBRACE')
        body = []
        while self.current_token() and self.current_token().type != 'RBRACE':
            stmt = self.parse_statement()
            if stmt:
                body.append(stmt)
        self.expect('RBRACE')
        
        return ForNode('for', variable, iterable, body)
    
    def parse_parallel(self) -> ParallelNode:
        """Parse parallel execution block."""
        self.expect('KEYWORD')  # 'parallel'
        self.expect('LBRACE')
        
        tasks = []
        while self.current_token() and self.current_token().type != 'RBRACE':
            stmt = self.parse_statement()
            if stmt:
                tasks.append(stmt)
        self.expect('RBRACE')
        
        return ParallelNode('parallel', tasks)
    
    def parse_execute(self) -> ExecuteNode:
        """Parse execute statement."""
        self.expect('KEYWORD')  # 'execute'
        self.expect('LPAREN')
        function = self.parse_expression()
        
        args = []
        while self.current_token() and self.current_token().type == 'COMMA':
            self.advance()
            args.append(self.parse_expression())
        
        self.expect('RPAREN')
        
        # Consume optional semicolon
        if self.current_token() and self.current_token().type == 'SEMICOLON':
            self.advance()
        
        return ExecuteNode('execute', function, args)
    
    def parse_assignment(self) -> AssignmentNode:
        """Parse assignment statement."""
        name = self.expect('IDENTIFIER').value
        self.expect('OPERATOR')  # '='
        value = self.parse_expression()
        if self.current_token() and self.current_token().type == 'SEMICOLON':
            self.advance()
        return AssignmentNode('assignment', name, value)
    
    def parse_expression(self) -> ASTNode:
        """Parse expression with operator precedence."""
        return self.parse_comparison()
    
    def parse_comparison(self) -> ASTNode:
        """Parse comparison operators."""
        left = self.parse_addition()
        
        while self.current_token() and self.current_token().type == 'OPERATOR' and self.current_token().value in ('==', '!=', '<', '>', '<=', '>='):
            op = self.current_token().value
            self.advance()
            right = self.parse_addition()
            left = BinaryOpNode('binary_op', op, left, right)
        
        return left
    
    def parse_addition(self) -> ASTNode:
        """Parse addition and subtraction."""
        left = self.parse_multiplication()
        
        while self.current_token() and self.current_token().type == 'OPERATOR' and self.current_token().value in ('+', '-'):
            op = self.current_token().value
            self.advance()
            right = self.parse_multiplication()
            left = BinaryOpNode('binary_op', op, left, right)
        
        return left
    
    def parse_multiplication(self) -> ASTNode:
        """Parse multiplication and division."""
        left = self.parse_unary()
        
        while self.current_token() and self.current_token().type == 'OPERATOR' and self.current_token().value in ('*', '/', '%'):
            op = self.current_token().value
            self.advance()
            right = self.parse_unary()
            left = BinaryOpNode('binary_op', op, left, right)
        
        return left
    
    def parse_unary(self) -> ASTNode:
        """Parse unary operators."""
        token = self.current_token()
        
        if token and token.type == 'PROBABILISTIC':
            self.advance()
            return ProbabilisticNode('probabilistic', self.parse_primary())
        
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        """Parse primary expressions."""
        token = self.current_token()
        
        if not token:
            raise SyntaxError("Unexpected end of input")
        
        # Execute as an expression
        if token.type == 'KEYWORD' and token.value.lower() == 'execute':
            return self.parse_execute()
        
        # Vector
        if token.type == 'VECTOR':
            self.advance()
            return VectorNode('vector', token.value)
        
        # Number
        if token.type == 'NUMBER':
            self.advance()
            return NumberNode('number', float(token.value))
        
        # String
        if token.type == 'STRING':
            self.advance()
            return StringNode('string', token.value)
        
        # List literal
        if token.type == 'LBRACKET':
            self.advance()
            elements = []
            while self.current_token() and self.current_token().type != 'RBRACKET':
                elements.append(self.parse_expression())
                if self.current_token() and self.current_token().type == 'COMMA':
                    self.advance()
            self.expect('RBRACKET')
            return ListNode('list', elements)
        
        # Identifier
        if token.type == 'IDENTIFIER':
            self.advance()
            return IdentifierNode('identifier', token.value)
        
        # Parenthesized expression
        if token.type == 'LPAREN':
            self.advance()
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr
        
        raise SyntaxError(f"Unexpected token: {token.type} = {token.value}")


# ============================================================================
# ENHANCED VECTOR CLASS WITH GPU SUPPORT
# ============================================================================

class AIVector:
    """Semantic vector with AI-centric operations and GPU support."""
    
    def __init__(self, values: List[float], use_gpu: bool = False):
        self.use_gpu = use_gpu and HAS_GPU
        if self.use_gpu:
            self.values = cp.array(values)
        else:
            self.values = np.array(values)
        self.confidence = 1.0
    
    def to_cpu(self):
        """Convert GPU array to CPU array."""
        if self.use_gpu:
            return AIVector(cp.asnumpy(self.values).tolist(), use_gpu=False)
        return self
    
    def to_gpu(self):
        """Convert CPU array to GPU array."""
        if HAS_GPU and not self.use_gpu:
            return AIVector(self.values.tolist(), use_gpu=True)
        return self
        
    def __repr__(self):
        vals = cp.asnumpy(self.values) if self.use_gpu else self.values
        device = "GPU" if self.use_gpu else "CPU"
        return f"âŸ¨{', '.join(f'{v:.3f}' for v in vals)}âŸ© ({device}, conf: {self.confidence:.2f})"
    
    def similarity(self, other: 'AIVector') -> float:
        """Compute cosine similarity."""
        if self.use_gpu:
            dot = cp.dot(self.values, other.values)
            norm = cp.linalg.norm(self.values) * cp.linalg.norm(other.values)
            result = dot / norm if norm > 0 else 0.0
            return float(cp.asnumpy(result))
        else:
            dot = np.dot(self.values, other.values)
            norm = np.linalg.norm(self.values) * np.linalg.norm(other.values)
            return float(dot / norm if norm > 0 else 0.0)
    
    def combine(self, other: 'AIVector') -> 'AIVector':
        """Combine two vectors (weighted average)."""
        combined = (self.values + other.values) / 2
        if self.use_gpu:
            result = AIVector(cp.asnumpy(combined).tolist(), use_gpu=True)
        else:
            result = AIVector(combined.tolist(), use_gpu=False)
        result.confidence = min(self.confidence, other.confidence)
        return result
    
    def tensor_product(self, other: 'AIVector') -> 'AIVector':
        """Simplified tensor product (outer product flattened)."""
        if self.use_gpu:
            outer = cp.outer(self.values, other.values).flatten()
            return AIVector(cp.asnumpy(outer).tolist(), use_gpu=True)
        else:
            outer = np.outer(self.values, other.values).flatten()
            return AIVector(outer.tolist(), use_gpu=False)


class ProbabilisticValue:
    """Value with uncertainty."""
    
    def __init__(self, mean: float, std: float = 0.1):
        self.mean = mean
        self.std = std
        
    def sample(self) -> float:
        """Sample from the distribution."""
        return random.gauss(self.mean, self.std)
    
    def __repr__(self):
        return f"~N({self.mean:.3f}, Ïƒ={self.std:.3f})"
    
    def __float__(self):
        return self.mean


# ============================================================================
# CONTROL FLOW EXCEPTIONS
# ============================================================================

class BreakException(Exception):
    """Exception to handle break statements."""
    pass

class ContinueException(Exception):
    """Exception to handle continue statements."""
    pass

class ReturnException(Exception):
    """Exception to handle return statements."""
    def __init__(self, value):
        self.value = value


# ============================================================================
# ENHANCED INTERPRETER
# ============================================================================

class Interpreter:
    """Executes AI++ Abstract Syntax Tree with enhanced features."""
    
    def __init__(self, use_gpu: bool = False, max_workers: int = 4):
        self.variables = {}
        self.functions = {}
        self.execution_stats = defaultdict(int)
        self.use_gpu = use_gpu and HAS_GPU
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def execute(self, nodes: List[ASTNode]) -> Any:
        """Execute a list of AST nodes."""
        result = None
        for node in nodes:
            result = self.evaluate(node)
        return result
    
    def evaluate(self, node: ASTNode) -> Any:
        """Evaluate a single AST node."""
        if node.type == 'vector':
            return AIVector(node.values, use_gpu=self.use_gpu)
        
        elif node.type == 'number':
            return node.value
        
        elif node.type == 'string':
            return node.value
        
        elif node.type == 'list':
            return [self.evaluate(elem) for elem in node.elements]
        
        elif node.type == 'identifier':
            if node.name not in self.variables:
                raise NameError(f"Undefined variable: {node.name}")
            return self.variables[node.name]
        
        elif node.type == 'binary_op':
            return self.evaluate_binary_op(node)
        
        elif node.type == 'probabilistic':
            value = self.evaluate(node.expression)
            if isinstance(value, (int, float)):
                return ProbabilisticValue(float(value))
            return value
        
        elif node.type == 'execute':
            return self.evaluate_execute(node)
        
        elif node.type == 'assignment':
            value = self.evaluate(node.value)
            self.variables[node.name] = value
            return value
        
        elif node.type == 'function_def':
            self.functions[node.name] = node
            return None
        
        elif node.type == 'if':
            return self.evaluate_if(node)
        
        elif node.type == 'while':
            return self.evaluate_while(node)
        
        elif node.type == 'for':
            return self.evaluate_for(node)
        
        elif node.type == 'break':
            raise BreakException()
        
        elif node.type == 'continue':
            raise ContinueException()
        
        elif node.type == 'return':
            value = self.evaluate(node.value) if node.value else None
            raise ReturnException(value)
        
        elif node.type == 'parallel':
            return self.evaluate_parallel(node)
        
        else:
            raise RuntimeError(f"Unknown node type: {node.type}")
    
    def evaluate_binary_op(self, node: BinaryOpNode) -> Any:
        """Evaluate binary operations."""
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)
        
        # Handle probabilistic values
        if isinstance(left, ProbabilisticValue):
            left = left.mean
        if isinstance(right, ProbabilisticValue):
            right = right.mean
        
        # Handle vectors
        if isinstance(left, AIVector) and isinstance(right, AIVector):
            if node.operator == '+':
                return left.combine(right)
            elif node.operator == '*':
                if left.use_gpu:
                    return AIVector((left.values * right.values).get().tolist(), use_gpu=True)
                else:
                    return AIVector((left.values * right.values).tolist(), use_gpu=False)
        
        # Numeric operations
        if node.operator == '+':
            return left + right
        elif node.operator == '-':
            return left - right
        elif node.operator == '*':
            return left * right
        elif node.operator == '/':
            return left / right if right != 0 else float('inf')
        elif node.operator == '%':
            return left % right
        elif node.operator == '==':
            return abs(left - right) < 0.0001 if isinstance(left, float) else left == right
        elif node.operator == '!=':
            return abs(left - right) >= 0.0001 if isinstance(left, float) else left != right
        elif node.operator == '<':
            return left < right
        elif node.operator == '>':
            return left > right
        elif node.operator == '<=':
            return left <= right
        elif node.operator == '>=':
            return left >= right
        
        raise RuntimeError(f"Unknown operator: {node.operator}")
    
    def evaluate_execute(self, node: ExecuteNode) -> Any:
        """Execute a function call."""
        self.execution_stats['executions'] += 1
        
        # Get function name
        if isinstance(node.function, IdentifierNode):
            func_name = node.function.name
        else:
            func_name = str(node.function)
        
        # Evaluate arguments
        args = [self.evaluate(arg) for arg in node.args]
        
        # Built-in functions
        if func_name == 'print':
            output = ' '.join(str(arg) for arg in args)
            print(output)
            return output
        
        elif func_name == 'similarity':
            if len(args) >= 2 and isinstance(args[0], AIVector) and isinstance(args[1], AIVector):
                return args[0].similarity(args[1])
            raise TypeError("similarity requires two vectors")
        
        elif func_name == 'optimize':
            # Simulate optimization
            if args:
                value = args[0]
                if isinstance(value, (int, float)):
                    return value * 1.1 + random.uniform(-0.1, 0.1)
                return value
            return None
        
        elif func_name == 'measure':
            # Return mock performance metrics
            return {
                'latency': random.uniform(10, 100),
                'accuracy': random.uniform(0.85, 0.99),
                'cost': random.uniform(0.01, 1.0),
                'device': 'GPU' if self.use_gpu else 'CPU'
            }
        
        # FILE I/O OPERATIONS
        elif func_name == 'read_file':
            if args:
                filepath = str(args[0])
                try:
                    with open(filepath, 'r') as f:
                        return f.read()
                except Exception as e:
                    print(f"Error reading file: {e}")
                    return None
            raise TypeError("read_file requires a filepath")
        
        elif func_name == 'write_file':
            if len(args) >= 2:
                filepath = str(args[0])
                content = str(args[1])
                try:
                    with open(filepath, 'w') as f:
                        f.write(content)
                    return True
                except Exception as e:
                    print(f"Error writing file: {e}")
                    return False
            raise TypeError("write_file requires filepath and content")
        
        elif func_name == 'read_json':
            if args:
                filepath = str(args[0])
                try:
                    with open(filepath, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"Error reading JSON: {e}")
                    return None
            raise TypeError("read_json requires a filepath")
        
        elif func_name == 'write_json':
            if len(args) >= 2:
                filepath = str(args[0])
                data = args[1]
                try:
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                    return True
                except Exception as e:
                    print(f"Error writing JSON: {e}")
                    return False
            raise TypeError("write_json requires filepath and data")
        
        # NETWORK/API OPERATIONS
        elif func_name == 'http_get':
            if not HAS_REQUESTS:
                print("Error: requests library not installed")
                return None
            if args:
                url = str(args[0])
                try:
                    response = requests.get(url, timeout=10)
                    return {
                        'status_code': response.status_code,
                        'text': response.text,
                        'headers': dict(response.headers)
                    }
                except Exception as e:
                    print(f"Error making HTTP request: {e}")
                    return None
            raise TypeError("http_get requires a URL")
        
        elif func_name == 'http_post':
            if not HAS_REQUESTS:
                print("Error: requests library not installed")
                return None
            if len(args) >= 2:
                url = str(args[0])
                data = args[1]
                try:
                    response = requests.post(url, json=data, timeout=10)
                    return {
                        'status_code': response.status_code,
                        'text': response.text,
                        'headers': dict(response.headers)
                    }
                except Exception as e:
                    print(f"Error making HTTP request: {e}")
                    return None
            raise TypeError("http_post requires URL and data")
        
        # GPU OPERATIONS
        elif func_name == 'to_gpu':
            if args and isinstance(args[0], AIVector):
                return args[0].to_gpu()
            return args[0] if args else None
        
        elif func_name == 'to_cpu':
            if args and isinstance(args[0], AIVector):
                return args[0].to_cpu()
            return args[0] if args else None
        
        elif func_name == 'gpu_available':
            return HAS_GPU
        
        # UTILITY FUNCTIONS
        elif func_name == 'range':
            if len(args) == 1:
                return list(range(int(args[0])))
            elif len(args) == 2:
                return list(range(int(args[0]), int(args[1])))
            elif len(args) == 3:
                return list(range(int(args[0]), int(args[1]), int(args[2])))
            raise TypeError("range requires 1-3 arguments")
        
        elif func_name == 'len':
            if args:
                return len(args[0])
            raise TypeError("len requires an argument")
        
        # User-defined functions
        elif func_name in self.functions:
            func_def = self.functions[func_name]
            
            # Create new scope
            old_vars = self.variables.copy()
            
            # Bind parameters
            for param, arg in zip(func_def.params, args):
                self.variables[param] = arg
            
            # Execute function body
            result = None
            try:
                for stmt in func_def.body:
                    result = self.evaluate(stmt)
            except ReturnException as e:
                result = e.value
            
            # Restore scope
            self.variables = old_vars
            
            return result
        
        else:
            raise NameError(f"Unknown function: {func_name}")
    
    def evaluate_if(self, node: IfNode) -> Any:
        """Evaluate if statement."""
        condition = self.evaluate(node.condition)
        
        # Handle probabilistic conditions
        if isinstance(condition, ProbabilisticValue):
            condition = condition.sample() > 0.5
        
        if condition:
            return self.execute(node.then_branch)
        elif node.else_branch:
            return self.execute(node.else_branch)
        
        return None
    
    def evaluate_while(self, node: WhileNode) -> Any:
        """Evaluate while loop."""
        result = None
        max_iterations = 100000  # Safety limit
        iterations = 0
        
        while iterations < max_iterations:
            condition = self.evaluate(node.condition)
            
            # Handle probabilistic conditions
            if isinstance(condition, ProbabilisticValue):
                condition = condition.sample() > 0.5
            
            if not condition:
                break
            
            try:
                for stmt in node.body:
                    result = self.evaluate(stmt)
            except BreakException:
                break
            except ContinueException:
                continue
            
            iterations += 1
        
        if iterations >= max_iterations:
            print("Warning: While loop exceeded maximum iterations")
        
        return result
    
    def evaluate_for(self, node: ForNode) -> Any:
        """Evaluate for loop."""
        iterable = self.evaluate(node.iterable)
        result = None
        
        if not hasattr(iterable, '__iter__'):
            raise TypeError(f"Cannot iterate over {type(iterable)}")
        
        for item in iterable:
            self.variables[node.variable] = item
            
            try:
                for stmt in node.body:
                    result = self.evaluate(stmt)
            except BreakException:
                break
            except ContinueException:
                continue
        
        return result
    
    def evaluate_parallel(self, node: ParallelNode) -> List[Any]:
        """Evaluate tasks in parallel using threads."""
        def execute_task(task):
            try:
                return self.evaluate(task)
            except Exception as e:
                return f"Error: {e}"
        
        futures = [self.executor.submit(execute_task, task) for task in node.tasks]
        results = [future.result() for future in futures]
        return results
    
    def get_stats(self) -> Dict:
        """Get execution statistics."""
        stats = dict(self.execution_stats)
        stats['gpu_enabled'] = self.use_gpu
        stats['gpu_available'] = HAS_GPU
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def run_aiplusplus(source_code: str, verbose: bool = False, use_gpu: bool = False) -> Any:
    """Run AI++ source code and return the result."""
    try:
        # Lexical analysis
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        if verbose:
            print("=== TOKENS ===")
            for token in tokens:
                print(f"{token.type:15} {token.value}")
            print()
        
        # Parsing
        parser = Parser(tokens)
        ast = parser.parse()
        
        if verbose:
            print("=== AST ===")
            for node in ast:
                print(node)
            print()
        
        # Interpretation
        interpreter = Interpreter(use_gpu=use_gpu)
        result = interpreter.execute(ast)
        
        if verbose:
            print("=== EXECUTION STATS ===")
            for key, value in interpreter.get_stats().items():
                print(f"{key}: {value}")
            print()
        
        interpreter.cleanup()
        return result
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("AI++ Programming Language Interpreter - ENHANCED")
    print("=" * 70)
    print()
    print("New Features:")
    print("  âœ“ While/For loops")
    print("  âœ“ File I/O operations")
    print("  âœ“ Network/API calls")
    print("  âœ“ Multi-threaded execution")
    print(f"  {'âœ“' if HAS_GPU else 'âœ—'} GPU acceleration")
    print()
    
    # Example: Loops
    print("=" * 70)
    print("Example 1: For Loop")
    print("=" * 70)
    code1 = """
for (i in execute(range, 5)) {
    execute(print, "Iteration:", i);
}
"""
    run_aiplusplus(code1)
    
    # Example: While Loop
    print("\n" + "=" * 70)
    print("Example 2: While Loop")
    print("=" * 70)
    code2 = """
counter = 0;
while (counter < 3) {
    execute(print, "Counter:", counter);
    counter = counter + 1;
}
"""
    run_aiplusplus(code2)
    
    # Example: File I/O
    print("\n" + "=" * 70)
    print("Example 3: File I/O")
    print("=" * 70)
    code3 = """
execute(write_file, "test.txt", "Hello from AI++");
content = execute(read_file, "test.txt");
execute(print, "File content:", content);
"""
    run_aiplusplus(code3)
    
    print("\n" + "=" * 70)
    print("Enhanced AI++ is ready!")
    print("=" * 70)
