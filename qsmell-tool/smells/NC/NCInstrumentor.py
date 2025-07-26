import ast

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