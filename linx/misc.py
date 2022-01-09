"""
Miscellaneous functions
"""
import os
from pathlib import Path


def get_version():
    """
    Get the current installed version of Linx.
    """

    return "0.0.8"


def get_tmp_path():
    """
    Get the tmp path where temporary data for tests get stored.
    """
    path = Path(f"{os.getenv('HOME')}/tmp_linx_dev")
    path.mkdir(exist_ok=True)

    return path


def clean_tmp(
    path=get_tmp_path(),
):
    """
    Clean the tmp folder.

    Parameters:
        path: Path-like object
    """
    path.mkdir(exist_ok=True)

    matches = list(path.rglob("*"))
    for match in matches:
        if not match.is_dir():
            match.unlink()

    descending_matches = sorted(
        matches,
        key=lambda path: len(path.parents),
        reverse=True
    )

    # Delete more nested folders first.
    for match in descending_matches:
        if match.is_dir():
            match.rmdir()
