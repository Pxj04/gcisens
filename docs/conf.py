project = "gcisens"
author = "Szymon Sniegowski"
copyright = "2026, Szymon Sniegowski"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

html_theme = "furo"
html_title = "gcisens"
