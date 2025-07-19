"""
Setup
"""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from environment variable or default
import os
version = os.environ.get('GITHUB_REF_NAME', '0.1.0')
if version.startswith('v'):
    version = version[1:]  # Remove 'v' prefix

setuptools.setup(
    name="graphica",
    version=version,
    author="Edderic Ugaddan",
    author_email="edderic@gmail.com",
    description="Tools for Probabilistic Graphical Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edderic/graphica",
    project_urls={
        "Bug Tracker": "https://github.com/edderic/graphica/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=["graphica"],
    python_requires=">=3.6",
)
