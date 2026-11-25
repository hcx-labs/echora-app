"""Increment the Navidrome plugin's own manifest version for a release."""

import argparse
import json
from pathlib import Path
import re


def increment_version(version: str, component: str) -> str:
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError(f"Invalid plugin version: {version!r}")
    major, minor, patch = map(int, version.split("."))
    if component == "major":
        major, minor, patch = major + 1, 0, 0
    elif component == "minor":
        minor, patch = minor + 1, 0
    elif component == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid increment: {component!r}")
    return f"{major}.{minor}.{patch}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("increment", choices=("patch", "minor", "major"))
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parents[2] / "navidrome/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = increment_version(manifest["version"], args.increment)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest["version"])


if __name__ == "__main__":
    main()
