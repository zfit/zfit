#!/usr/bin/env python

#  Copyright (c) 2025 zfit

from __future__ import annotations

import atexit
import hashlib

# disable gpu for TensorFlow
import os
import pickle
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pygit2
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU for TensorFlow

import zfit

project_dir = Path(__file__).parents[1]
# sys.path.insert(0, str(project_dir))

# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.


# caching for plots, todo: factore out and move somewhere else
class PlotCache:
    """A caching mechanism for plot generation that checks file hashes."""

    def __init__(self, cache_dir: Path):
        """Initialize the plot cache.

        Args:
            cache_dir: Directory to store cache metadata
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "plot_cache.pkl"
        self.cache_data = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache data from disk."""
        if self.cache_file.exists():
            try:
                with self.cache_file.open("rb") as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                return {}
        return {}

    def _save_cache(self):
        """Save cache data to disk."""
        with self.cache_file.open("wb") as f:
            pickle.dump(self.cache_data, f)

    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of a file."""
        if not file_path.exists():
            return ""

        hash_md5 = hashlib.md5()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def should_regenerate(self, source_file: Path, output_files: list[Path]) -> bool:
        """Check if plots should be regenerated.

        Args:
            source_file: The source file that generates the plots
            output_files: List of output image files

        Returns:
            True if plots should be regenerated, False otherwise
        """
        # Check if any output files are missing
        if not all(output_file.exists() for output_file in output_files):
            return True

        # Get current hash of source file
        current_hash = self._get_file_hash(source_file)

        # Check if hash has changed
        cache_key = str(source_file)
        if cache_key not in self.cache_data:
            return True

        cached_hash = self.cache_data[cache_key].get("hash", "")
        return current_hash != cached_hash

    def mark_generated(self, source_file: Path, output_files: list[Path]):
        """Mark plots as generated and update cache.

        Args:
            source_file: The source file that generates the plots
            output_files: List of output image files that were generated
        """
        current_hash = self._get_file_hash(source_file)
        cache_key = str(source_file)

        self.cache_data[cache_key] = {"hash": current_hash, "output_files": [str(f) for f in output_files]}

        self._save_cache()

    def cached_generation(self, source_file: Path, output_files: list[Path], generate_func: Callable[[], Any]) -> bool:
        """Execute generation function only if cache is invalid.

        Args:
            source_file: The source file that generates the plots
            output_files: List of output image files
            generate_func: Function to call to generate the plots

        Returns:
            True if plots were generated, False if cache was used
        """
        if self.should_regenerate(source_file, output_files):
            print(f"Regenerating plots for {source_file.name}")
            generate_func()
            self.mark_generated(source_file, output_files)
            return True
        else:
            print(f"Using cached plots for {source_file.name}")
            return False


needs_sphinx = "3.0.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    # sphinx_autodoc_typehints must be imported after napoleon to properly work.
    # See https://github.com/agronholm/sphinx-autodoc-typehints/issues/15
    "jupyter_sphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinxcontrib.youtube",
    "sphinx_panels",
    "myst_nb",
    "sphinx_togglebutton",
]

panels_add_bootstrap_css = False  # for sphinx_panel, use custom css from theme, not bootstrap

# releases_github_path = "zfit/zfit"  # TODO: use releases or similar?
# releases_document_name = "../CHANGELOG.rst"

# nb_execution_mode = "force"  # use if needed and cache should be ignored
nb_execution_mode = "off"
# nb_execution_mode = "cache"
if nb_execution_mode == "cache":
    jupyter_cache_path = project_dir.joinpath("docs", ".cache", "myst-nb")
    jupyter_cache_path.mkdir(parents=True, exist_ok=True)
    nb_execution_cache_path = str(jupyter_cache_path)


myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

bibtex_bibfiles = ["refs.bib"]  # str(project_dir.joinpath("docs", "refs.bib"))]
bibtex_default_style = "plain"

# Import plot cache utility


# run the generate_pdf_plots.py script to generate the pdf plots
docsdir = project_dir / "docs"
plotscript = docsdir / "utils" / "generate_pdf_plots.py"
minimizerscript = docsdir / "utils" / "generate_minimizer_plots.py"
plot_output_dir = docsdir / "_static" / "plots"
plot_output_dir.mkdir(parents=True, exist_ok=True)

# Initialize plot cache
cache_dir = docsdir / ".cache" / "plots"
plot_cache = PlotCache(cache_dir)

# PDF plots generation with caching
pdf_images_dir = docsdir / "images" / "_generated" / "pdfs"
pdf_images_dir.mkdir(parents=True, exist_ok=True)


def generate_pdf_plots():
    """Generate PDF plots."""
    subprocess.run([sys.executable, str(plotscript)], check=True, stdout=subprocess.PIPE)


# Check if PDF plots need regeneration
pdf_output_files = list(pdf_images_dir.glob("*.png"))
if plot_cache.should_regenerate(plotscript, pdf_output_files):
    print("Regenerating PDF plots...")
    generate_pdf_plots()
    plot_cache.mark_generated(plotscript, pdf_output_files)
else:
    print("Using cached PDF plots")

# Minimizer plots generation with caching
minimizer_output_dir = docsdir / "_static" / "minimizer_plots"
minimizer_output_dir.mkdir(parents=True, exist_ok=True)
minimizer_images_dir = docsdir / "images" / "_generated" / "minimizers"
minimizer_images_dir.mkdir(parents=True, exist_ok=True)


def generate_minimizer_plots():
    """Generate minimizer plots."""
    subprocess.run([sys.executable, str(minimizerscript)], check=True, stdout=subprocess.PIPE)


# Check if minimizer plots need regeneration
minimizer_output_files = list(minimizer_images_dir.glob("*.png")) + list(minimizer_images_dir.glob("*.gif"))
if plot_cache.should_regenerate(minimizerscript, minimizer_output_files):
    print("Regenerating minimizer plots...")
    generate_minimizer_plots()
    plot_cache.mark_generated(minimizerscript, minimizer_output_files)
else:
    print("Using cached minimizer plots")
# Temporarily disabled for faster build
zfit_tutorials_path = project_dir.joinpath("docs", "_tmp", "zfit-tutorials")
atexit.register(lambda path=zfit_tutorials_path: shutil.rmtree(path))
pygit2.clone_repository("https://github.com/zfit/zfit-tutorials", zfit_tutorials_path)

# todo: do we need it and connect?
# zfit_physics_path = project_dir / "docs" / "_tmp" / "zfit-physics"
# atexit.register(lambda path=zfit_physics_path: shutil.rmtree(path))
# pygit2.clone_repository("https://github.com/zfit/zfit-physics", zfit_physics_path)

# Temporarily disabled for faster build
zfit_images_path = docsdir / "images"
docs_images_path = docsdir / "_static" / "images"
atexit.register(lambda path=docs_images_path: shutil.rmtree(path))
docs_images_path.mkdir(parents=True, exist_ok=True)
shutil.copytree(zfit_images_path, docs_images_path, dirs_exist_ok=True)

nb_execution_in_temp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:

source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb", ".myst": "myst-nb"}

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "zfit"
copyright = zfit.__copyright__
author = zfit.__author__

# The version info for the project you're documenting, acts as replacement
# for |version| and |release|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
version = zfit.__version__.split("+")[0]
# The full version, including alpha/beta/rc tags.
release = zfit.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".cache", "README.md", "README.rst"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# add whitespaces to the internal commands. Maybe move to preprocessing?
rst_epilog = """
.. |wzw| unicode:: U+200B
   :trim:

"""
# ..  replace:: |wzw|
#
# .. |@docend| replace:: |wzw|
# """
with Path(project_dir / "utils/api/argdocs.yaml").open() as replfile:
    replacements = yaml.load(replfile, Loader=yaml.Loader)
for replacement_key in replacements:
    rst_epilog += f"""
.. |@doc:{replacement_key}| replace:: |wzw|

.. |@docend:{replacement_key}| replace:: |wzw|
"""
print("replacements", replacements)
with Path("hyperlinks.txt").open() as hyperlinks:
    rst_epilog += hyperlinks.read()

# makes the jupyter extension executable
jupyter_sphinx_thebelab_config = {
    # 'bootstrap': False,
    # 'requestKernel': True,
    "binderOptions": {
        "repo": "zfit/zfit-tutorials",
        "binderUrl": "https://mybinder.org",
        "repoProvider": "github",
    },
}

html_favicon = "images/zfit-favicon.png"

# -- Napoleon settings ---------------------------------------------

using_numpy_style = False  # False -> google style
napoleon_google_docstring = not using_numpy_style
napoleon_numpy_docstring = using_numpy_style
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- sphinx_autodoc_typehints settings ---------------------------------------------

# if True, set typing.TYPE_CHECKING to True to enable “expensive” typing imports
set_type_checking_flag = True
# if True, class names are always fully qualified (e.g. module.for.Class). If False, just the class
# name displays (e.g. Class)
typehints_fully_qualified = False
# (default: False): If False, do not add ktype info for undocumented parameters. If True, add stub documentation for
# undocumented parameters to be able to add type info.
always_document_param_types = False
# (default: True): If False, never add an :rtype: directive. If True, add the :rtype: directive if no existing :rtype:
# is found.
typehints_document_rtype = True

# -- autodoc settings ---------------------------------------------

# also doc __init__ docstrings
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "show-inheritance": True,
    "inherited-members": True,
}
autodoc_inherit_docstrings = False

# -- autosummary settings ---------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True

# -- sphinx.ext.todo settings ---------------------------------------------
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

# Theme options are theme-specific and customize the look and feel of a
# theme further.

html_theme_options = {
    "logo": {
        "image_light": "images/zfit-logo_400x168.png",
        "image_dark": "images/zfit-logo-light_400x168.png",
    },
    "github_url": "https://github.com/zfit/zfit",
    "use_edit_page_button": True,
    "navigation_depth": 2,
    "search_bar_text": "Search zfit...",
    "navigation_with_keys": True,
    # "search_bar_position": "sidebar",
    "icon_links": [{}],  # temporary fix for https://github.com/pydata/pydata-sphinx-theme/issues/1220
    # "repository_url": "https://github.com/zfit/zfit",  # adding jupyter book somehow?
    # "repository_branch": "develop",
    # "path_to_docs": "docs",
    "collapse_navigation": True,
    # "sticky_navigation": True,
    # "navigation_depth": 4,
    "header_links_before_dropdown": 7,
}

html_context = {
    "github_user": "zfit",
    "github_repo": "zfit",
    "github_version": "develop",
    "doc_path": "docs",
}

# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "zfitdoc"

# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto, manual, or own class]).
latex_documents = [
    (master_doc, "zfit.tex", "zfit Documentation", "zfit", "manual"),
]

# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "zfit", "zfit Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "zfit",
        "zfit Documentation",
        author,
        "zfit",
        "One line description of project.",
        "Miscellaneous",
    ),
]
# cross reference
default_role = "py:obj"
primary_domain = "py"
# nitpicky = True  # warn if cross-references are missing
# nitpick_ignore = [
#     ("py:class", "tensorflow.keras.losses.Loss"),
# ]


intersphinx_mapping = {
    # 'numdifftools': ('https://numdifftools.readthedocs.io/en/latest/index.html', None),
    "tensorflow": (
        "https://www.tensorflow.org/api_docs/python",
        "https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tf2_py_objects.inv",
    ),
    "tensorflow_probability": (
        "https://www.tensorflow.org/probability/api_docs/python",
        "https://raw.githubusercontent.com/GPflow/tensorflow-intersphinx/master/tfp_py_objects.inv",
    ),
    "uproot": ("https://uproot.readthedocs.io/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
