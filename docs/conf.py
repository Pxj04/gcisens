project = "gcisens"
author = "Szymon Sniegowski and Adrianna Świder"
copyright = "2026, gcisens contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
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
html_theme_options = {
    "source_repository": "https://github.com/Pxj04/gcisens/",
    "source_branch": "main",
    "source_directory": "docs/",
}
