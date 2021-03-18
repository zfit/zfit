#  Copyright (c) 2021 zfit

import argparse
import os
import re

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
    replacements = yaml.load(replfile, Loader=yaml.Loader)

# Replace the target string
auto_end = r'|@docend|'
auto_start = r'|@docstart|'
for filepath in cfg.files:
    if not filepath.endswith('.py'):
        continue
    with open(filepath, 'r') as file:
        filedata = file.read()

    infile = False
    for param, replacement in replacements.items():
        replacement = replacement.rstrip('\n')
        auto_param = r'|@doc:{}|'.format(param)
        param_mod = f'{auto_start}{auto_param}{auto_end}'
        matches = re.findall(auto_start.replace('|', r'\|')
                             + auto_param.replace('|', r'\|')
                             + r".*?"
                             + auto_end.replace('|', r'\|'), filedata, re.DOTALL)
        if not matches:
            continue
        infile = True

        replacement_mod = f'{auto_start}{auto_param}{replacement}{auto_end}'

        for match in matches:
            filedata = filedata.replace(match, replacement_mod)

    # Write the file out again
    if infile:
        if cfg.dry:
            print(f'Not writing to {filepath}, dry run.')
        else:
            with open(filepath, 'w') as file:
                file.write(filedata)
            print(f'Written to {filepath}.')
