import pytest

import molflux.splits.config
from molflux.splits.info import SplittingStrategyInfo


def test_description_required():
    """That a 'description' field is required on initialisation."""
    required = "description"
    with pytest.raises(
        TypeError,
        match=f"required positional argument: * {required!r}",
    ):
        SplittingStrategyInfo()  # type: ignore[call-arg]


def test_to_dict():
    """That can dump the SplittingStrategyInfo object as dict."""
    info = SplittingStrategyInfo(description="test")
    info_dict = info.to_dict()
    assert "description" in info_dict
    assert info_dict.get("description") == "test"


def test_write_to_directory_creates_expected_file(tmp_path):
    """That can dump the SplittingStrategyInfo object as file."""
    info = SplittingStrategyInfo(description="test")
    directory = tmp_path / "info"

    directory.mkdir()
    assert len(list(directory.glob("**/*"))) == 0

    info.write_to_directory(directory=str(directory))
    assert len(list(directory.glob("**/*"))) == 1

    expected_info_filename = (
        directory / molflux.splits.config.SPLITTING_STRATEGY_INFO_FILENAME
    )
    assert expected_info_filename.is_file()
