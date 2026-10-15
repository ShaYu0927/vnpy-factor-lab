"""Keep directory prefixes consistent across project-owned source files."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_modules_use_directory_prefixes() -> None:
    invalid = []
    for directory, directories, filenames in os.walk(PROJECT_ROOT / "vnpy"):
        directories[:] = [
            name
            for name in directories
            if not name.startswith(".")
            and name not in {"__pycache__", "venv", "env", "logs", "data"}
        ]
        folder = Path(directory)
        for name in filenames:
            if not name.endswith(".py") or name in {"__init__.py", "__main__.py"}:
                continue
            if not name.startswith(f"{folder.name}_"):
                invalid.append((folder / name).relative_to(PROJECT_ROOT).as_posix())

    assert not invalid, f"Modules must start with their directory name: {invalid}"


def test_configuration_and_frontend_sources_use_directory_prefixes() -> None:
    paths = list((PROJECT_ROOT / "config").glob("*.json"))
    paths.extend((PROJECT_ROOT / "web" / "src").glob("*.js"))
    paths.extend((PROJECT_ROOT / "web" / "src").glob("*.css"))
    invalid = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in paths
        if not path.name.startswith(f"{path.parent.name}_")
    ]

    assert not invalid, f"Files must start with their directory name: {invalid}"
