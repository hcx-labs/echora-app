"""The code lives in src/ and imports as `worker` (setuptools reads the rest from pyproject.toml)."""

from setuptools import find_packages, setup

setup(
    package_dir={"worker": "src"},
    packages=["worker", *(f"worker.{name}" for name in find_packages("src"))],
)
