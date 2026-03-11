# Configuration file for the Sphinx documentation builder.

import os
import sys
import time

# -- Path setup --------------------------------------------------------------

# Add project root so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "so4gp"
author = "Dickson Owuor"
copyright = f"{time.localtime().tm_year}, Dickson Owuor"

# Full version
release = "0.2.5"

# Short version
version = "0.2"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "myst_parser",
    "sphinx_github_changelog",
]

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Autodoc settings --------------------------------------------------------

autoclass_content = "both"
autodoc_preserve_defaults = True

# Remove duplicate constructor documentation
def remove_lines_before_parameters(app, what, name, obj, options, lines):
    if what == "class":
        first_idx = next(
            (i for i, line in enumerate(lines) if line.startswith(":param")),
            None,
        )
        if first_idx is not None:
            lines[:] = lines[first_idx:]


def setup(app):
    app.connect("autodoc-process-docstring", remove_lines_before_parameters)


# -- HTML output -------------------------------------------------------------

html_theme = "furo"

html_title = "so4gp — Gradual Pattern Mining Algorithms"

html_baseurl = "https://so4gp.readthedocs.io/"

html_copy_source = False

html_theme_options = {
    # "description": "A collection of gradual pattern mining algorithms and tools",  ## Not supported by Furo
    "source_repository": "https://github.com/owuordickson/sogp_pypi",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": ["view", "edit"],
}

html_static_path = ["_static"]
html_css_files = []

# -- EPUB --------------------------------------------------------------------

# epub_show_urls = "footnote"