"""The code lives in src/ and imports as `backend` (setuptools reads the rest from pyproject.toml)."""

from setuptools import find_packages, setup

setup(
    package_dir={"backend": "src"},
    packages=["backend", *(f"backend.{name}" for name in find_packages("src"))],
)
