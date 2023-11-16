from molflux.datasets.load import load_from_dicts

representative_dataset_name = "esol"


def test_returns_dataset_dict():
    """That loading from a dict returns a Dataset."""
    name = representative_dataset_name
    config = {
        "name": name,
        "config": {},
    }
    dataset_map = load_from_dicts([config])
    assert isinstance(dataset_map, dict)
