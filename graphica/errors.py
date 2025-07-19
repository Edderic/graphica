"""
Error classes for the graphica package.
"""


class GraphicaException(Exception):
    """Base exception class for graphica package."""


class ArgumentError(GraphicaException):
    """Exception raised when invalid arguments are provided."""
