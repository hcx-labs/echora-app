"""The code lives in src/ and imports as `core` (setuptools reads the rest from pyproject.toml)."""

from setuptools import find_packages, setup

setup(
    package_dir={"core": "src"},
    packages=["core", *(f"core.{name}" for name in find_packages("src"))],
)
