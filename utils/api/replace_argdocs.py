#  Copyright (c) 2021 zfit

import argparse
import os

import yaml

here = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description='Apply cuts to ntuple',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("files", nargs='*',
                    help='Files to be processed.')

parser.add_argument("--dry", action="store_true",
                    help='Dry run WITHOUT replacing.')

cfg = parser.parse_args()

with open(here + '/argdocs.yaml') as replfile:
    replacements = yaml.load(replfile)

# Replace the target string
auto_end = '|@docend|'
auto_start = '|@docstart|'
for filepath in cfg.files:
    with open(filepath, 'r') as file:
        filedata = file.read()

    for param, replacement in replacements.items():
        auto_param = f'|@doc:{param}|'
        param_mod = f'{auto_start}{auto_param}{auto_end}'
        replacement_mod = f'{auto_start}{auto_param}{replacement}{auto_end}'
        filedata = filedata.replace(param_mod, replacement_mod)

    # Write the file out again
    if cfg.dry:
        print(f'Not writing to {filepath}, dry run.')
    else:
        with open(filepath, 'w') as file:
            file.write(filedata)
