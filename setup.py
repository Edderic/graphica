"""
Setup
"""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="linx",
    version="0.0.3",
    author="Edderic Ugaddan",
    author_email="edderic@gmail.com",
    description="Tools for Probabilistic Graphical Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edderic/linx",
    project_urls={
        "Bug Tracker": "https://github.com/edderic/linx/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=["linx"],
    python_requires=">=3.6",
)
