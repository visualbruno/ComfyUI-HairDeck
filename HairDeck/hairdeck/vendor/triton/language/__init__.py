"""Placeholders for triton.language: see hairdeck/vendor/triton."""
from .. import _Any


def __getattr__(name):
    return _Any(f'triton.language.{name}')
