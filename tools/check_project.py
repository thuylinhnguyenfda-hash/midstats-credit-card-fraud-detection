"""Lightweight review checks; does not load the CSV, fit models or score test."""
from pathlib import Path
import ast
import json
import csv
import nbformat

root = Path(__file__).resolve().parents[1]
notebook = nbformat.read(root / 'Banking_Fraud_Project.ipynb', as_version=4)
nbformat.validate(notebook)
code_cells = [c for c in notebook.cells if c.cell_type == 'code']
code = '\n'.join(c.source for c in code_cells)

for old_input in ['sample_all44.parquet', 'scan_manifest.json', '_full.csv', 'category_split']:
    assert old_input not in code, f'Old all-dataset EDA dependency: {old_input}'

eda = [c for c in code_cells if c.metadata.get('stage') == 'train_eda']
assert eda, 'Train EDA stage markers are missing.'
for cell in eda:
    for forbidden in ['transactions', 'heldout', 'X_va', 'X_te', 'test_ready', 'validation_ready', 'pd.read_csv', 'pd.read_parquet']:
        assert forbidden not in cell.source, f'Held-out/cached dependency in train EDA: {forbidden}'

split_last = max(i for i, c in enumerate(notebook.cells) if c.metadata.get('stage') == 'split_only')
eda_first = min(i for i, c in enumerate(notebook.cells) if c.metadata.get('stage') == 'train_eda')
assert split_last < eda_first, 'Split must precede EDA.'
assert 'data = train.sample(n=min(200_000, len(train)), random_state=42)' in code
assert 'RUN_FINAL_TEST = False' in code, 'Review default should keep final test disabled.'

for cell in code_cells:
    tree = ast.parse('\n'.join(line for line in cell.source.splitlines() if not line.startswith('%')))
    has_test_load = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == 'load' and node.args
                        and isinstance(node.args[0], ast.Constant) and node.args[0].value == 'test'
                        for node in ast.walk(tree))
    if has_test_load:
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.If)
        assert ast.dump(tree.body[0].test) == ast.dump(ast.Name(id='RUN_FINAL_TEST', ctx=ast.Load()))
    assert not any(o.output_type == 'error' for o in cell.outputs), 'Saved error output found.'

with (root / 'data/data_dictionary.csv').open(newline='') as file:
    reader = csv.DictReader(file)
    assert set(reader.fieldnames) == {'column', 'group', 'description', 'unit_or_encoding'}
    assert len(list(reader)) == 44

info_path = root / 'results/run_info.json'
if info_path.exists():
    info = json.loads(info_path.read_text())
    assert info['eda_scope'] == 'train_only'
    assert not info['final_test_evaluated'], 'This review package should not contain a new real-test run.'

print('PASS: split precedes train-only EDA; old global inputs absent; review test gate and metadata are valid.')
