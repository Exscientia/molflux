from molflux.datasets.load import load_from_yaml


def test_from_yaml_returns_metrics(fixture_path_to_assets):
    path = fixture_path_to_assets / "config.yml"
    datasets = load_from_yaml(path=path)
    assert len(datasets) == 1
    assert "dataset-0" in datasets
    assert isinstance(datasets, dict)
