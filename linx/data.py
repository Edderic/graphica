"""
Data module
"""
from uuid import uuid4

import pandas as pd

from .errors import ArgumentError


class ParquetData:
    """
    Meant to load data in a lazy way to not max out RAM when there are many
    variables.

    Parameters:
        data: An object that implements
            - read_parquet
            - to_parquet
    """
    def __init__(self, data, storage_folder, name=None):
        if name is None:
            self.name = uuid4()
        else:
            self.name = name

        self.storage_folder = storage_folder
        self.path = storage_folder / str(self.name)

        if self.path.exists():
            raise ArgumentError(f"{self.path} already exists")

        self.write(data)

    def get_storage_folder(self):
        """
        Return: Path-like object
        """
        return self.storage_folder

    def read(self):
        """
        Read the data.

        Returns: data
        """
        return pd.read_parquet(self.path, engine='pyarrow')

    def write(self, data):
        """
        Write the data.
        """
        data.to_parquet(self.path, engine='pyarrow')
