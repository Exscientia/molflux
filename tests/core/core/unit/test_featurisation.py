from typing import Any, Dict

import pytest

import datasets
import molflux.modelzoo
from molflux.core import featurise_dataset, replay_dataset_featurisation, save_model


@pytest.fixture()
def fixture_sample_dataset() -> datasets.Dataset:
    """A sample training dataset. Smiles from TDC lipo"""
    return datasets.Dataset.from_dict(
        {
            "canonical_smiles": [
                "CCCCCCCC(=O)OCN1C(=O)C(NC1=O)(c2ccccc2)c3ccccc3",
                "CCN(=O)=O",
                "CCN(CC(=C)C)c1c(cc(cc1N(=O)=O)C(F)(F)F)N(=O)=O",
                "c1cc(c(c(c1Cl)Cl)Cl)Cl",
                "CCCC(C)(COC(=O)N)COC(=O)N",
            ],
            "y": [1, 2, 3, 4, 5],
        },
    )


@pytest.fixture()
def fixture_sample_featurisation_metadata() -> Dict[str, Any]:
    """A sample featurisation metadata for a dataset with 'canonical_smiles'"""
    return {
        "version": 1,
        "config": [
            {
                "column": "canonical_smiles",
                "representations": [
                    {
                        "name": "character_count",
                        "as": "character_count",
                    },
                    {
                        "name": "character_count",
                        "as": "my_alias",
                        "config": {
                            "tag": "c_count",
                        },
                        "presets": {},
                    },
                ],
            },
        ],
    }


def test_featurise_dataset_with_unsupported_featurisation_metadata_version_raises(
    fixture_sample_dataset,
):
    """That an error is raised if attempting to featurise a dataset with an
    unsupported version of featurisation metadata schema."""

    dataset = fixture_sample_dataset
    featurisation_metadata = {"version": 0, "config": {}}

    with pytest.raises(NotImplementedError, match="version"):
        featurise_dataset(dataset, featurisation_metadata=featurisation_metadata)


def test_featurise_dataset(
    fixture_sample_dataset,
    fixture_sample_featurisation_metadata,
):
    """That can featurise a dataset using well-formed featurisation metadata."""

    dataset = fixture_sample_dataset
    featurisation_metadata = fixture_sample_featurisation_metadata

    featurised_dataset = featurise_dataset(
        dataset,
        featurisation_metadata=featurisation_metadata,
    )
    assert len(featurised_dataset.column_names) > len(dataset.column_names)
    assert "my_alias" in featurised_dataset.column_names


def test_featurise_dataset_with_map_kwargs(
    fixture_sample_dataset,
    fixture_sample_featurisation_metadata,
):
    """That can featurise a dataset using well-formed featurisation metadata and use map_kwargs."""

    dataset = fixture_sample_dataset
    featurisation_metadata = fixture_sample_featurisation_metadata

    featurised_dataset = featurise_dataset(
        dataset,
        featurisation_metadata=featurisation_metadata,
        batch_size=2,
    )
    assert len(featurised_dataset.column_names) > len(dataset.column_names)
    assert "my_alias" in featurised_dataset.column_names


def test_replay_featurisation(
    tmp_path,
    fixture_sample_dataset,
    fixture_sample_featurisation_metadata,
):
    """That can replay featurisation for a given saved model."""

    dataset = fixture_sample_dataset
    featurisation_metadata = fixture_sample_featurisation_metadata
    featurised_dataset = featurise_dataset(
        dataset,
        featurisation_metadata=featurisation_metadata,
    )

    model = molflux.modelzoo.load_model(
        "linear_regressor",
        x_features=["my_alias", "character_count"],
        y_features=["y"],
    )
    model.train(featurised_dataset)
    model_path = save_model(
        model,
        path=tmp_path,
        featurisation_metadata=featurisation_metadata,
    )

    replayed_dataset = replay_dataset_featurisation(dataset, model_path=model_path)
    assert replayed_dataset.column_names == featurised_dataset.column_names


def test_replay_featurisation_with_map_kwargs(
    tmp_path,
    fixture_sample_dataset,
    fixture_sample_featurisation_metadata,
):
    """That can replay featurisation for a given saved model with map_kwargs."""

    dataset = fixture_sample_dataset
    featurisation_metadata = fixture_sample_featurisation_metadata
    featurised_dataset = featurise_dataset(
        dataset,
        featurisation_metadata=featurisation_metadata,
    )

    model = molflux.modelzoo.load_model(
        "linear_regressor",
        x_features=["my_alias", "character_count"],
        y_features=["y"],
    )
    model.train(featurised_dataset)
    model_path = save_model(
        model,
        path=tmp_path,
        featurisation_metadata=featurisation_metadata,
    )

    replayed_dataset = replay_dataset_featurisation(
        dataset,
        model_path=model_path,
        batch_size=2,
    )
    assert replayed_dataset.column_names == featurised_dataset.column_names
