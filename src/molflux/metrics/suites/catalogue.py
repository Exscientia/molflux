import difflib
from pathlib import Path
from typing import Dict, List

SUITES_CATALOGUE: Dict[str, Path] = {}


def get_suite_path(suite_name: str) -> Path:
    """Returns the path to the given suite from the catalogue."""
    suite_path = SUITES_CATALOGUE.get(suite_name)

    if suite_path is None:
        msg = f"Suite {suite_name!r} is not available."
        similar = difflib.get_close_matches(suite_name, SUITES_CATALOGUE.keys())
        if similar:
            msg += f" You might be looking for one of these: {similar}"
        raise NotImplementedError(msg)

    if not suite_path.is_file():
        raise FileNotFoundError(f"Could not find suite definition file: {suite_path!r}")

    return suite_path


def list_suites() -> List[str]:
    """List all available suites."""
    suite_names = sorted(SUITES_CATALOGUE.keys())
    return suite_names


def register_suite(suite_path: Path) -> None:
    """Registers a suite in the catalogue."""
    file_type = suite_path.suffix
    if file_type not in {".yml", ".yaml"}:
        raise ValueError(f"Invalid suite file format: {file_type!r} (expected .yml)")

    suite_name = suite_path.stem
    if suite_name not in SUITES_CATALOGUE:
        SUITES_CATALOGUE[suite_name] = suite_path
