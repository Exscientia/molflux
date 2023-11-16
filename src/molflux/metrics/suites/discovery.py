from pathlib import Path

from molflux.metrics.suites.catalogue import register_suite

_CWD: Path = Path(__file__).parent.resolve()


def import_suites() -> None:
    """Load all available suites in the catalogue."""
    paths = (
        path
        for path in _CWD.iterdir()
        if path.suffix == ".yml" or path.suffix == ".yaml"
    )
    for path in paths:
        register_suite(suite_path=path)
