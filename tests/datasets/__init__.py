import os
from typing import Any


def set_test_cache() -> None:
    """Sets a dedicated huggingface cache directory for tests."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets/pytest")
    os.environ["HF_DATASETS_CACHE"] = cache_dir


def mock_load_dataset_to_force_redownload():
    """Mocks the 'datasets.load_dataset()' function called by
    'molflux.datasets.load_dataset()' to force the redownload of assets,
    keeping tests isolated.

    This is because there is no easy way to stop huggingface from downloading
    datasets globally.
    """
    import molflux.datasets

    original_function = molflux.datasets.load.datasets.load_dataset

    def mock_load_dataset(*args: Any, **kwargs: Any):  # type:ignore[no-untyped-def]
        kwargs.update(
            {"download_mode": "force_redownload", "verification_mode": "no_checks"},
        )
        return original_function(*args, **kwargs)

    molflux.datasets.load.datasets.load_dataset = mock_load_dataset


set_test_cache()
mock_load_dataset_to_force_redownload()
