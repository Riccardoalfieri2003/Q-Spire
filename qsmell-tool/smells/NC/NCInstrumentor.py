import ast
"""
class NCInstrumentor(ast.NodeTransformer):

    def visit_Assign(self, node):
        # Keep original assign
        new_nodes = [node]

        # Check if it's backend creation
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == 'AerSimulator':
                # backend = AerSimulator()
                target_name = node.targets[0].id
                patch_call = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id='patch_backend_run', ctx=ast.Load()),
                        args=[ast.Name(id=target_name, ctx=ast.Load())],
                        keywords=[]
                    )
                )
                new_nodes.append(patch_call)

        # Check if it's circuit creation
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == 'init_circuit':
                # qc = init_circuit(...)
                target_name = node.targets[0].id
                patch_call = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id='t_assign_parameters', ctx=ast.Load()),
                        args=[ast.Name(id=target_name, ctx=ast.Load())],
                        keywords=[]
                    )
                )
                new_nodes.append(patch_call)

        return new_nodes


    def visit_Call(self, node):
        # We want to detect calls like qc.run(...) or qc.assign_parameters(...)
        # node.func is an Attribute
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ('run', 'assign_parameters'):
                # Attempt to get caller variable name
                if isinstance(node.func.value, ast.Name):
                    caller_name = node.func.value.id
                else:
                    caller_name = "<unknown>"

                # We wrap the original call with the logging helper
                # Build ast for: log_and_call_run(caller_name, lineno, col_start, col_end, caller_obj, *args, **kwargs)
                new_node = ast.Call(
                    func=ast.Name(id=f'log_and_call_{method_name}', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=caller_name),
                        ast.Constant(value=node.lineno),
                        ast.Constant(value=node.col_offset),
                        ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                        node.func.value,  # the caller object, e.g. qc
                    ] + node.args,
                    keywords=node.keywords,
                )
                return ast.copy_location(new_node, node)

        return self.generic_visit(node)

"""


# NCInstrumentor.py
import ast
import ast
import ast
"""
class NCInstrumentor(ast.NodeTransformer):

    def visit_Assign(self, node):
        new_nodes = [node]

        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                func_name = node.value.func.id

                if func_name == 'AerSimulator':
                    target_name = node.targets[0].id
                    # backend = patch_backend_run(backend)
                    patch_node = ast.Assign(
                        targets=[ast.Name(id=target_name, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id='patch_backend_run', ctx=ast.Load()),
                            args=[ast.Name(id=target_name, ctx=ast.Load())],
                            keywords=[]
                        )
                    )
                    new_nodes.append(patch_node)

                elif func_name == 'QuantumCircuit':
                    target_name = node.targets[0].id
                    # qc = t_assign_parameters(qc)
                    patch_node = ast.Assign(
                        targets=[ast.Name(id=target_name, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id='t_assign_parameters', ctx=ast.Load()),
                            args=[ast.Name(id=target_name, ctx=ast.Load())],
                            keywords=[]
                        )
                    )
                    new_nodes.append(patch_node)

        return new_nodes
"""

"""
    Instruments method calls:
      backend.run(...) → log_and_call_run(...)
      qc.assign_parameters(...) → log_and_call_assign_parameters(...)
    """


"""
class NCInstrumentor(ast.NodeTransformer):

    def visit_Call(self, node):
        self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            caller_obj = node.func.value

            if method_name in ('run', 'assign_parameters'):
                # Try to get the first argument variable name
                first_arg_name = None
                if node.args and isinstance(node.args[0], ast.Name):
                    first_arg_name = node.args[0].id

                new_node = ast.Call(
                    func=ast.Name(id=f'log_and_call_{method_name}', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=method_name),
                        ast.Constant(value=node.lineno),
                        ast.Constant(value=node.col_offset),
                        ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                        caller_obj
                    ] + node.args + [
                        ast.Constant(value=first_arg_name)
                    ],
                    keywords=node.keywords,
                )
                return ast.copy_location(new_node, node)

        return node
"""


















"""
class NCInstrumentor(ast.NodeTransformer):
   
       def visit_Call(self, node):
        self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            caller_obj = node.func.value

            if method_name == 'run':
                # For run method, we need to be careful about argument order
                if node.args and isinstance(node.args[0], ast.Name):
                    first_arg_name = node.args[0].id
                else:
                    first_arg_name = None

                new_node = ast.Call(
                    func=ast.Name(id='log_and_call_run', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=method_name),
                        ast.Constant(value=node.lineno),
                        ast.Constant(value=node.col_offset),
                        ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                        caller_obj,  # backend object
                        node.args[0] if node.args else ast.Name(id='None', ctx=ast.Load()),  # circuit object
                        ast.Constant(value=first_arg_name)  # circuit variable name
                    ] + (node.args[1:] if len(node.args) > 1 else []),
                    keywords=node.keywords,
                )
                return ast.copy_location(new_node, node)

            elif method_name == 'assign_parameters':
                # Handle assign_parameters as before
                # Original args without the parameters
                other_args = node.args[1:] if len(node.args) > 1 else []
                
                new_node = ast.Call(
                    func=ast.Name(id=f'log_and_call_{method_name}', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=method_name),
                        ast.Constant(value=node.lineno),
                        ast.Constant(value=node.col_offset),
                        ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                        caller_obj,
                        ast.Constant(value=first_arg_name),
                        node.args[0] if node.args else ast.Constant(value=None)  # parameters
                    ] + other_args,
                    keywords=node.keywords,
                )
            else:
                # Original run method instrumentation
                new_node = ast.Call(
                    func=ast.Name(id=f'log_and_call_{method_name}', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=method_name),
                        ast.Constant(value=node.lineno),
                        ast.Constant(value=node.col_offset),
                        ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                        caller_obj,
                        ast.Constant(value=first_arg_name)
                    ] + node.args,
                    keywords=node.keywords,
                )
            
            return ast.copy_location(new_node, node)






















            if method_name == 'run':
                # For run method, we need to be careful about argument order
                if node.args and isinstance(node.args[0], ast.Name):
                    first_arg_name = node.args[0].id
                else:
                    first_arg_name = None

                new_node = ast.Call(
                    func=ast.Name(id='log_and_call_run', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=method_name),
                        ast.Constant(value=node.lineno),
                        ast.Constant(value=node.col_offset),
                        ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                        caller_obj,  # backend object
                        node.args[0] if node.args else ast.Name(id='None', ctx=ast.Load()),  # circuit object
                        ast.Constant(value=first_arg_name)  # circuit variable name
                    ] + (node.args[1:] if len(node.args) > 1 else []),
                    keywords=node.keywords,
                )
                return ast.copy_location(new_node, node)

            # For assign_parameters, we need to handle the parameters argument carefully
            if method_name == 'assign_parameters':
                # Original args without the parameters
                other_args = node.args[1:] if len(node.args) > 1 else []
                
                new_node = ast.Call(
                    func=ast.Name(id=f'log_and_call_{method_name}', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=method_name),
                        ast.Constant(value=node.lineno),
                        ast.Constant(value=node.col_offset),
                        ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                        caller_obj,
                        ast.Constant(value=first_arg_name),
                        node.args[0] if node.args else ast.Constant(value=None)  # parameters
                    ] + other_args,
                    keywords=node.keywords,
                )
            else:
                # Original run method instrumentation
                new_node = ast.Call(
                    func=ast.Name(id=f'log_and_call_{method_name}', ctx=ast.Load()),
                    args=[
                        ast.Constant(value=method_name),
                        ast.Constant(value=node.lineno),
                        ast.Constant(value=node.col_offset),
                        ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                        caller_obj,
                        ast.Constant(value=first_arg_name)
                    ] + node.args,
                    keywords=node.keywords,
                )
            
            return ast.copy_location(new_node, node)
        return node
"""


"""
class NCInstrumentor(ast.NodeTransformer):
   
    def visit_Call(self, node):
        self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            caller_obj = node.func.value

            if method_name in ('run', 'assign_parameters'):
                # Initialize first_arg_name with None as default
                first_arg_name = None
                
                # Try to get the first argument variable name if it exists
                if node.args and isinstance(node.args[0], ast.Name):
                    first_arg_name = node.args[0].id

                if method_name == 'run':
                    new_node = ast.Call(
                        func=ast.Name(id='log_and_call_run', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=method_name),
                            ast.Constant(value=node.lineno),
                            ast.Constant(value=node.col_offset),
                            ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                            caller_obj,  # backend object
                            node.args[0] if node.args else ast.Name(id='None', ctx=ast.Load()),  # circuit object
                            ast.Constant(value=first_arg_name)  # circuit variable name
                        ] + (node.args[1:] if len(node.args) > 1 else []),
                        keywords=node.keywords,
                    )
                elif method_name == 'assign_parameters':
                    # Handle assign_parameters case
                    other_args = node.args[1:] if len(node.args) > 1 else []
                    
                    new_node = ast.Call(
                        func=ast.Name(id=f'log_and_call_{method_name}', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=method_name),
                            ast.Constant(value=node.lineno),
                            ast.Constant(value=node.col_offset),
                            ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                            caller_obj,
                            ast.Constant(value=first_arg_name),
                            node.args[0] if node.args else ast.Constant(value=None)  # parameters
                        ] + other_args,
                        keywords=node.keywords,
                    )
                
                return ast.copy_location(new_node, node)

        return node
"""




class NCInstrumentor(ast.NodeTransformer):
   
    def visit_Call(self, node):
        self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            caller_obj = node.func.value

            if method_name in ('run', 'assign_parameters'):
                # Initialize first_arg_name with None as default
                first_arg_name = None
                
                # For assign_parameters, we need to get the circuit name from the caller object
                if method_name == 'assign_parameters':
                    # Check if the caller is an attribute access (like circuit.assign_parameters)
                    if isinstance(caller_obj, ast.Name):
                        first_arg_name = caller_obj.id
                    # Or if it's a method call on an attribute (like circuit.method().assign_parameters)
                    elif isinstance(caller_obj, ast.Attribute):
                        if isinstance(caller_obj.value, ast.Name):
                            first_arg_name = caller_obj.value.id
                
                # For run method, get from first argument
                elif method_name == 'run' and node.args and isinstance(node.args[0], ast.Name):
                    first_arg_name = node.args[0].id

                if method_name == 'run':
                    new_node = ast.Call(
                        func=ast.Name(id='log_and_call_run', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=method_name),
                            ast.Constant(value=node.lineno),
                            ast.Constant(value=node.col_offset),
                            ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                            caller_obj,  # backend object
                            node.args[0] if node.args else ast.Name(id='None', ctx=ast.Load()),  # circuit object
                            ast.Constant(value=first_arg_name)  # circuit variable name
                        ] + (node.args[1:] if len(node.args) > 1 else []),
                        keywords=node.keywords,
                    )
                elif method_name == 'assign_parameters':
                    other_args = node.args[1:] if len(node.args) > 1 else []
                    
                    new_node = ast.Call(
                        func=ast.Name(id=f'log_and_call_{method_name}', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=method_name),
                            ast.Constant(value=node.lineno),
                            ast.Constant(value=node.col_offset),
                            ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                            caller_obj,  # circuit object
                            ast.Constant(value=first_arg_name),  # circuit variable name
                            node.args[0] if node.args else ast.Constant(value=None)  # parameters
                        ] + other_args,
                        keywords=node.keywords,
                    )
                
                return ast.copy_location(new_node, node)

        return node







"""def _preserve_circuit_names(circuits):
    if isinstance(circuits, list):
        for c in circuits:
            if not hasattr(c, '_original_circuit'):
                c._original_circuit = c
    elif not hasattr(circuits, '_original_circuit'):
        circuits._original_circuit = circuits
    return circuits"""
"""
class NCInstrumentor(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self._parent = None

    def visit(self, node):
        # Track parent nodes for context
        node.parent = getattr(self, '_parent', None)
        old_parent = self._parent
        self._parent = node
        result = super().visit(node)
        self._parent = old_parent
        return result

    def visit_Call(self, node):
        self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            caller_obj = node.func.value

            # Handle transpile() calls first
            if method_name == 'transpile':
                # Mark circuits before transpiling
                new_args = [
                    self._wrap_transpile_arg(node.args[0]),
                    *[self.visit(arg) for arg in node.args[1:]]
                ]
                new_node = ast.Call(
                    func=ast.Attribute(
                        value=self.visit(caller_obj),
                        attr='transpile',
                        ctx=ast.Load()
                    ),
                    args=new_args,
                    keywords=[self.visit(k) for k in node.keywords]
                )
                return ast.copy_location(new_node, node)

            if method_name in ('run', 'assign_parameters'):
                first_arg_name = self._get_circuit_name(method_name, node, caller_obj)

                if method_name == 'run':
                    new_node = ast.Call(
                        func=ast.Name(id='log_and_call_run', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=method_name),
                            ast.Constant(value=node.lineno),
                            ast.Constant(value=node.col_offset),
                            ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                            caller_obj,  # backend object
                            node.args[0] if node.args else ast.Name(id='None', ctx=ast.Load()),
                            ast.Constant(value=first_arg_name)
                        ] + (node.args[1:] if len(node.args) > 1 else []),
                        keywords=node.keywords,
                    )
                elif method_name == 'assign_parameters':
                    new_node = ast.Call(
                        func=ast.Name(id='log_and_call_assign_parameters', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=method_name),
                            ast.Constant(value=node.lineno),
                            ast.Constant(value=node.col_offset),
                            ast.Constant(value=getattr(node, 'end_col_offset', node.col_offset + 1)),
                            caller_obj,
                            ast.Constant(value=first_arg_name),
                            node.args[0] if node.args else ast.Constant(value=None)
                        ] + (node.args[1:] if len(node.args) > 1 else []),
                        keywords=node.keywords,
                    )
                return ast.copy_location(new_node, node)

        return node

    def _get_circuit_name(self, method_name, node, caller_obj):
        if method_name == 'assign_parameters':
            if isinstance(caller_obj, ast.Name):
                return caller_obj.id
            elif isinstance(caller_obj, ast.Attribute) and isinstance(caller_obj.value, ast.Name):
                return caller_obj.value.id
            elif (isinstance(node.parent, ast.ListComp) and 
                  isinstance(caller_obj, ast.Call) and
                  isinstance(caller_obj.func, ast.Attribute) and
                  caller_obj.func.attr == 'assign_parameters'):
                if isinstance(caller_obj.func.value, ast.Name):
                    return caller_obj.func.value.id
        elif method_name == 'run' and node.args and isinstance(node.args[0], ast.Name):
            return node.args[0].id
        return None

    def _wrap_transpile_arg(self, arg_node):
        return ast.Call(
            func=ast.Name(id='_preserve_circuit_names', ctx=ast.Load()),
            args=[self.visit(arg_node)],
            keywords=[]
        )
"""