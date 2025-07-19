# Graphica

Tools for Probabilistic Graphical Modeling.

# For maintainers:

There is a pre-commit hook that you can use.

```
ln -s git_hooks/pre-commit .git/hooks/
```

When you want to release a new version:

- Update the version number in setup.py
- Clean previous builds: `rm -rf dist/ build/ *.egg-info/`
- Build new distribution: `python setup.py sdist bdist_wheel`
- Upload to PyPI: `twine upload dist/*`
