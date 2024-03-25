#!/usr/bin/env python
#  Copyright (c) 2024 zfit
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import yaml

here = Path(os.path.realpath(__file__)).parent

parser = argparse.ArgumentParser(
    description="Replace arguments with central stored ones",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("files", nargs="*", help="Files to be processed.")

parser.add_argument("--dry", action="store_true", help="Dry run WITHOUT replacing.")

cfg = parser.parse_args()

with Path(here / "argdocs.yaml").open() as replfile:
    replacements = yaml.load(replfile, Loader=yaml.Loader)

# Replace the target string
# auto_end_old = r'|@docend|'
for filepath in cfg.files:
    if not filepath.endswith(".py"):
        continue
    with Path(filepath).open() as file:
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
                msg = f"Docstring formatting error," f" has more than one start until an end command: {match}"
                raise ValueError(msg)
            if match != replacement_mod:
                needs_replacement = True
                filedata = filedata.replace(match, replacement_mod)

    # Write the file out again
    replace_msg = "replaced docs" if needs_replacement else "docs already there"
    filename = filepath.split("/")[-1]
    if infile:
        if cfg.dry:
            pass
        elif needs_replacement:
            with Path(filepath).open("w") as file:
                file.write(filedata)
