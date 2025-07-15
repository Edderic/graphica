# Graphica

Tools for Probabilistic Graphical Modeling.

# For maintainers:
When you want to release a new version:

- Update the version number in setup.py
- Clean previous builds: `rm -rf dist/ build/ *.egg-info/`
- Build new distribution: `python setup.py sdist bdist_wheel`
- Upload to PyPI: `twine upload dist/*`
