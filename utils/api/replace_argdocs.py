#!/usr/bin/env python
#  Copyright (c) 2022 zfit

import argparse
import os
import re

import yaml

here = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(
    description="Replace arguments with central stored ones",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("files", nargs="*", help="Files to be processed.")

parser.add_argument("--dry", action="store_true", help="Dry run WITHOUT replacing.")

cfg = parser.parse_args()

with open(here + "/argdocs.yaml") as replfile:
    replacements = yaml.load(replfile, Loader=yaml.Loader)

# Replace the target string
# auto_end_old = r'|@docend|'
for filepath in cfg.files:
    if not filepath.endswith(".py"):
        continue
    with open(filepath) as file:
        filedata = file.read()

    infile = False
    needs_replacement = False
    for param, replacement in replacements.items():
        replacement = replacement.rstrip("\n")
        while replacement[:1] == " ":  # we want to remove the whitespace
            replacement = replacement[1:]
        auto_start = rf"|@doc:{param}|"
        auto_end = rf"|@docend:{param}|"
        matches = re.findall(
            auto_start.replace("|", r"\|") + r".*?" + auto_end.replace("|", r"\|"),
            filedata,
            re.DOTALL,
        )

        if not matches:
            continue
        infile = True

        replacement_mod = f"{auto_start} {replacement} {auto_end}"

        for match in matches:
            if auto_start in match[len(auto_start) :]:  # sanity check
                raise ValueError(
                    f"Docstring formatting error,"
                    f" has more than one start until an end command: {match}"
                )
            if match != replacement_mod:
                needs_replacement = True
                filedata = filedata.replace(match, replacement_mod)

    # Write the file out again
    replace_msg = "replaced docs" if needs_replacement else "docs already there"
    filename = filepath.split("/")[-1]
    if infile:
        if cfg.dry:
            print(
                f"Match in {filename}, {replace_msg}, not writing to {filepath}, dry run."
            )
        else:
            if needs_replacement:
                with open(filepath, "w") as file:
                    file.write(filedata)
                print(f"Modified {filename}.")
