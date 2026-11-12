"""Sphinx configuration for the LASSQD documentation."""

import sys
from pathlib import Path

# Document the checkout without installing it, so the docs build needs no mrh.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

project = "LASSQD"
author = "LASSQD developers"
release = "0.1.0"

extensions = [
    "myst_nb",
    "qiskit_sphinx_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
exclude_patterns = ["_build", "**/__pycache__", "**/.ipynb_checkpoints"]

# mrh needs native builds; lassqd.las only imports it at module level.
autodoc_mock_imports = ["mrh"]
autodoc_default_options = {"members": True, "inherited-members": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# The tutorial is committed with its outputs; running it needs mrh and Fulqrum.
nb_execution_mode = "off"
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
    "qiskit-addon-sqd": (
        "https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd/",
        None,
    ),
}

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
html_theme_options = {
    "source_repository": "https://github.com/joannaqw/lassqd_code/",
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_qiskit_ecosystem_member": False,
}
