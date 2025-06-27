import ast
import astor

class FixedTypeRenamer(ast.NodeTransformer):
    def __init__(self):
        self.var_types = {}

    def _infer_type(self, value):
        if isinstance(value, ast.Constant):
            if isinstance(value.value, int):
                return 'int'
            elif isinstance(value.value, str):
                return 'str'
            elif isinstance(value.value, float):
                return 'float'
            elif isinstance(value.value, bool):
                return 'bool'
        elif isinstance(value, ast.List):
            return 'list'
        elif isinstance(value, ast.Dict):
            return 'dict'
        elif isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                return f"{value.func.id}_var"
        return 'var'

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            inferred = self._infer_type(node.value)
            self.var_types[name] = inferred
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if node.id in self.var_types:
            node.id = self.var_types[node.id]
        return node

def clean_code(code: str) -> str:
    tree = ast.parse(code)
    renamer = FixedTypeRenamer()
    renamed = renamer.visit(tree)
    ast.fix_missing_locations(renamed)
    return astor.to_source(renamed)
