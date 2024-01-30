from typing import Any, Dict

import pytest

import datasets
from molflux.datasets import featurise_dataset


class MockRepresentation:
    """Implements the interfaces.Representation protocol.

    Here we define a single representation that simply return the input samples
    untouched. This can be used to simulate one-to-one representations.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def featurise(self, samples, **kwargs):
        return {self.name: samples}


class MockMultiRepresentation:
    """Implements the interfaces.Representation protocol.

    Here we define a single representation that returns n copies of the input
    samples untouched. This can be used to simulate one-to-many representations.
    """

    def __init__(self, name: str, n: int) -> None:
        self.name = name
        self.n = n

    def featurise(self, samples: Any, **kwargs: Any) -> Dict[str, Any]:
        return {f"{self.name}{i}": samples for i in range(self.n)}


@pytest.fixture(scope="module")
def fixture_mock_dataset() -> datasets.Dataset:
    data = {
        "a": [1, 2, 3, 4, 5],
        "b": [6, 7, 8, 9, 10],
    }
    dataset = datasets.Dataset.from_dict(data)
    return dataset


@pytest.fixture(scope="module")
def fixture_mock_dataset_dict() -> datasets.DatasetDict:
    ds_train = datasets.Dataset.from_dict({"a": [1, 2, 3, 4]})
    ds_test = datasets.Dataset.from_dict({"a": [1, 3, 4, 2]})
    return datasets.DatasetDict({"train": ds_train, "test": ds_test})


@pytest.fixture(scope="module")
def fixture_mock_dataset_dict_with_empty_splits() -> datasets.DatasetDict:
    ds_train = datasets.Dataset.from_dict({"a": []})
    ds_validation = datasets.Dataset.from_dict({"a": [1, 3, 4, 2]})
    ds_test = datasets.Dataset.from_dict({"a": []})
    return datasets.DatasetDict(
        {"train": ds_train, "validation": ds_validation, "test": ds_test},
    )


@pytest.fixture(scope="module")
def fixture_mock_one_to_one_representation():
    """A single representation that returns one arbitrary output."""
    return MockRepresentation("original_name_a")


@pytest.fixture(scope="module")
def fixture_mock_one_to_many_representation():
    """A single representation  that returns 3 arbitrary outputs."""
    return MockMultiRepresentation("original_name_a", n=3)


@pytest.fixture(scope="module")
def fixture_mock_representations():
    """A collection of representations that simulates an heterogeneous mix of
    representations including representations generating multiple features.
    """

    return [
        MockRepresentation("original_name_a"),
        MockMultiRepresentation("original_name_b", n=2),
        MockRepresentation("original_name_c"),
    ]


@pytest.mark.parametrize(
    "representations_fixture",
    [
        "fixture_mock_one_to_many_representation",
        "fixture_mock_representations",
    ],
)
def test_can_use_representation_or_representations(
    fixture_mock_dataset,
    representations_fixture,
    request,
):
    """That can featurise a dataset both providing a single representation (convenience)
    or a collection of representations (core)."""

    representations_fixture = request.getfixturevalue(representations_fixture)

    dataset = fixture_mock_dataset
    representations = representations_fixture
    column_to_featurise = dataset.column_names[0]
    assert featurise_dataset(
        dataset=dataset,
        column=column_to_featurise,
        representations=representations,
    )


def test_featurise_dataset_raises_on_invalid_source_column(
    fixture_mock_dataset,
    fixture_mock_representations,
):
    """That an error is raised if attempting to featurise a non-existent column."""
    dataset = fixture_mock_dataset
    representations = fixture_mock_representations

    with pytest.raises(KeyError, match=r"Feature .+? not in dataset"):
        featurise_dataset(
            dataset=dataset,
            column="non-existent-column",
            representations=representations,
        )


def test_featurise_dataset_adds_features(
    fixture_mock_dataset,
    fixture_mock_representations,
):
    """That the featurised dataset has additional columns with new features."""
    dataset = fixture_mock_dataset
    representations = fixture_mock_representations
    column_to_featurise = dataset.column_names[0]
    featurised_dataset = featurise_dataset(
        dataset=dataset,
        column=column_to_featurise,
        representations=representations,
    )
    assert len(featurised_dataset.column_names) > len(dataset.column_names)


@pytest.mark.parametrize(
    ("display_names", "expected_output_names"),
    [
        (
            # canonical format (nested array with one array of display names for each representation)
            [["custom_name_a"]],
            ["custom_name_a"],
        ),
        (
            # shorthand for single representation (worth to support as we allow users to pass a single representation to the function)
            ["custom_name_a"],
            ["custom_name_a"],
        ),
        (
            # apply None (fallback) template across all representations
            None,
            ["a::original_name_a"],
        ),
        (
            # canonical format, apply fallback template across all outputs of associated representation (same effect in this case)
            [[None]],
            ["a::original_name_a"],
        ),
        (
            # same but using shorthand for single representation
            [None],
            ["a::original_name_a"],
        ),
        (
            # apply custom template across all representations
            "{source_column}>>{feature_name}",
            ["a>>original_name_a"],
        ),
        (
            # canonical format, apply custom template across all outputs of associated representation (same effect in this case)
            [["{source_column}>>{feature_name}"]],
            ["a>>original_name_a"],
        ),
        (
            # shorthand for single representation
            ["{source_column}>>{feature_name}"],
            ["a>>original_name_a"],
        ),
    ],
)
def test_allowed_display_names_notations_for_one_to_one_representation_outputs(
    fixture_mock_dataset,
    fixture_mock_one_to_one_representation,
    display_names,
    expected_output_names,
):
    """That can assign custom display names to the featurisation outputs obtained
    using a single one-to-one representation.

    All display names formats shown above should be accepted and give the
    corresponding output.
    """
    dataset = fixture_mock_dataset
    representation = fixture_mock_one_to_one_representation
    column_to_featurise = dataset.column_names[0]
    featurised_dataset = featurise_dataset(
        dataset=dataset,
        column=column_to_featurise,
        representations=representation,
        display_names=display_names,
    )
    for display_name in expected_output_names:
        assert display_name in featurised_dataset.column_names


@pytest.mark.parametrize(
    ("display_names", "expected_output_names"),
    [
        (
            # canonical format (nested array with one array of display names for each representation)
            [["custom_name_a0", "custom_name_a1", "custom_name_a2"]],
            ["custom_name_a0", "custom_name_a1", "custom_name_a2"],
        ),
        (
            # shorthand for single representation
            ["custom_name_a0", "custom_name_a1", "custom_name_a2"],
            ["custom_name_a0", "custom_name_a1", "custom_name_a2"],
        ),
        (
            # canonical format (1 fallback to default template)
            [["custom_name_a0", None, "custom_name_a2"]],
            ["custom_name_a0", "a::original_name_a1", "custom_name_a2"],
        ),
        (
            # canonical format (2 fallbacks to default template)
            [["custom_name_a0", None, None]],
            ["custom_name_a0", "a::original_name_a1", "a::original_name_a2"],
        ),
        (
            # canonical format (all fallbacks to default template, explicit)
            [[None, None, None]],
            ["a::original_name_a0", "a::original_name_a1", "a::original_name_a2"],
        ),
        (
            # canonical format (all fallbacks to default template, implicit)
            [[None]],
            ["a::original_name_a0", "a::original_name_a1", "a::original_name_a2"],
        ),
        (
            # shorthand for single representation
            [None],
            ["a::original_name_a0", "a::original_name_a1", "a::original_name_a2"],
        ),
        (
            # apply None (fallback) template across all representations (only one here)
            None,
            ["a::original_name_a0", "a::original_name_a1", "a::original_name_a2"],
        ),
        (
            # apply custom template across all representations
            "{source_column}>>{feature_name}",
            ["a>>original_name_a0", "a>>original_name_a1", "a>>original_name_a2"],
        ),
        # ...
    ],
)
def test_allowed_display_names_notations_for_one_to_many_representation_outputs(
    fixture_mock_dataset,
    fixture_mock_one_to_many_representation,
    display_names,
    expected_output_names,
):
    """That can assign custom display names to the featurisation outputs obtained
    using a single one-to-many representation.

    All display names formats shown above should be accepted and give the
    corresponding output.
    """
    dataset = fixture_mock_dataset
    representation = fixture_mock_one_to_many_representation
    column_to_featurise = dataset.column_names[0]
    featurised_dataset = featurise_dataset(
        dataset=dataset,
        column=column_to_featurise,
        representations=representation,
        display_names=display_names,
    )
    for display_name in expected_output_names:
        assert display_name in featurised_dataset.column_names


@pytest.mark.parametrize(
    ("display_names", "expected_output_names"),
    [
        (
            # canonical format (nested array with one array of display names for each representation)
            [
                ["custom_name_a"],
                ["custom_name_b0", "custom_name_b1"],
                ["custom_name_c"],
            ],
            ["custom_name_a", "custom_name_b0", "custom_name_b1", "custom_name_c"],
        ),
        (
            # canonical syntax, mix of custom display names and fallback templates
            [[None], ["custom_name_b0", None], ["custom_name_c"]],
            [
                "a::original_name_a",
                "custom_name_b0",
                "a::original_name_b1",
                "custom_name_c",
            ],
        ),
        (
            # canonical format (all fallback templates, explicit)
            [[None], [None, None], [None]],
            [
                "a::original_name_a",
                "a::original_name_b0",
                "a::original_name_b1",
                "a::original_name_c",
            ],
        ),
        (
            # canonical format (all fallback templates, using implicit syntax for multi-output feature)
            [[None], [None], [None]],
            [
                "a::original_name_a",
                "a::original_name_b0",
                "a::original_name_b1",
                "a::original_name_c",
            ],
        ),
        (
            # apply None (fallback) template across all representations
            None,
            [
                "a::original_name_a",
                "a::original_name_b0",
                "a::original_name_b1",
                "a::original_name_c",
            ],
        ),
        (
            # apply custom template across all representations
            "{source_column}>>{feature_name}",
            [
                "a>>original_name_a",
                "a>>original_name_b0",
                "a>>original_name_b1",
                "a>>original_name_c",
            ],
        ),
        # ...
    ],
)
def test_allowed_display_names_notations_for_representations_outputs(
    fixture_mock_dataset,
    fixture_mock_representations,
    display_names,
    expected_output_names,
):
    """That can assign custom display names to the featurisation outputs obtained
    using a single one-to-many representation.

    All display names formats shown above should be accepted for collections of
    representations and give the corresponding output.
    """
    dataset = fixture_mock_dataset
    representations = fixture_mock_representations
    column_to_featurise = dataset.column_names[0]
    featurised_dataset = featurise_dataset(
        dataset=dataset,
        column=column_to_featurise,
        representations=representations,
        display_names=display_names,
    )
    for display_name in expected_output_names:
        assert display_name in featurised_dataset.column_names


def test_using_shorthand_display_names_syntax_with_representations_raises(
    fixture_mock_dataset,
    fixture_mock_representations,
):
    """That using display names in the single-representation shorthand notation
    but when providing multiple representations raises an error.
    """
    dataset = fixture_mock_dataset
    representations = fixture_mock_representations
    column_to_featurise: str = dataset.column_names[0]
    shorthand_display_names = [
        "custom_name_a",
        "custom_name_b0",
        "custom_name_b1",
        "custom_name_c",
    ]
    with pytest.raises(ValueError, match="can only be used for single representations"):
        featurise_dataset(
            dataset=dataset,
            column=column_to_featurise,
            representations=representations,
            display_names=shorthand_display_names,  # type: ignore[arg-type]
        )


def test_using_unsupported_display_names_syntax_raises(
    fixture_mock_dataset,
    fixture_mock_representations,
):
    """That using display names in an unsupported syntax format raises an error."""
    dataset = fixture_mock_dataset
    representations = fixture_mock_representations
    column_to_featurise = dataset.column_names[0]
    mixed_notation_display_names = [
        "custom_name_a",
        ["custom_name_b0", "custom_name_b1"],
        "custom_name_c",
    ]
    with pytest.raises(ValueError, match="do not follow the accepted syntax"):
        featurise_dataset(
            dataset=dataset,
            column=column_to_featurise,
            representations=representations,
            display_names=mixed_notation_display_names,  # type: ignore[arg-type]
        )


def test_providing_not_enough_display_names_for_number_of_representations_used_raises(
    fixture_mock_dataset,
    fixture_mock_representations,
):
    """That not providing enough display names for the number of representations
    being used raises an error."""
    dataset = fixture_mock_dataset
    representations = fixture_mock_representations
    column_to_featurise = dataset.column_names[0]
    display_names = [
        ["custom_name_a"],
        [None],
    ]  # missing display names for third representation
    with pytest.raises(RuntimeError, match="Error parsing display names"):
        featurise_dataset(
            dataset=dataset,
            column=column_to_featurise,
            representations=representations,
            display_names=display_names,  # type: ignore[arg-type]
        )


def test_using_non_template_as_template_raises(
    fixture_mock_dataset,
    fixture_mock_representations,
):
    """That safeguards are in place against using a static string as template to
    be broadcasted across all output features. This would implicitly rename
    all outputs to the same static name, resulting in a single output feature."""
    dataset = fixture_mock_dataset
    representations = fixture_mock_representations
    column_to_featurise = dataset.column_names[0]
    mixed_notation_display_names = [
        ["custom_name_a"],
        ["not_a_template"],
        ["custom_name_c"],
    ]
    with pytest.raises(ValueError, match="Only template display names can be used"):
        featurise_dataset(
            dataset=dataset,
            column=column_to_featurise,
            representations=representations,
            display_names=mixed_notation_display_names,  # type: ignore[arg-type]
        )


def test_using_a_template_with_unknown_fields_raises(
    fixture_mock_dataset,
    fixture_mock_representations,
):
    """That setting a template requiring fields which are not made available
    by the backend templating context raises an error.
    """
    dataset = fixture_mock_dataset
    representations = fixture_mock_representations
    column_to_featurise = dataset.column_names[0]
    template = "{source_column}::{unsupported_key}"

    with pytest.raises(
        KeyError,
        match="Could not template the target string: 'unsupported_key'",
    ):
        featurise_dataset(
            dataset=dataset,
            column=column_to_featurise,
            representations=representations,
            display_names=template,
        )


def test_setting_display_name_already_in_use_warns(
    fixture_mock_dataset,
    fixture_mock_representations,
):
    """That setting a display name that is the same as the name of an existent
    column in the dataset raises a warning.

    This should be allowed to enable more complex / dynamic config driven
    featurisation workflows.
    """
    dataset = fixture_mock_dataset
    representations = fixture_mock_representations
    column_to_featurise = dataset.column_names[0]
    existing_column_name: str = dataset.column_names[1]
    display_names = [[existing_column_name], [None], [None]]

    # Check that the warning mentions the column being overwritten
    with pytest.warns(UserWarning, match=dataset.column_names[1]):
        featurise_dataset(
            dataset=dataset,
            column=column_to_featurise,
            representations=representations,
            display_names=display_names,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "dataset_dict",
    [
        "fixture_mock_dataset_dict",
        "fixture_mock_dataset_dict_with_empty_splits",
    ],
)
def test_can_featurise_dataset_dict(
    dataset_dict,
    fixture_mock_representations,
    request,
):
    """That can featurise a DatasetDict.

    We would generally advise featurising a single `Dataset` and then splitting
    it into a `DatasetDict`, but feturising a `DatasetDict` should technically
    be supported too.
    """

    dataset_dict = request.getfixturevalue(dataset_dict)

    representations = fixture_mock_representations

    column_to_featurise = dataset_dict.column_names["train"][0]
    featurised_dataset_dict = featurise_dataset(
        dataset=dataset_dict,
        column=column_to_featurise,
        representations=representations,
    )

    # check there is the same number of Datasets
    assert len(featurised_dataset_dict) == len(dataset_dict)

    # check that all Datasets have been featurised
    for split_name, featurised_datast in featurised_dataset_dict.items():
        assert len(featurised_datast.column_names) > len(
            dataset_dict[split_name].column_names,
        )


def test_raises_on_inconsistent_dataset_dict(fixture_mock_representations):
    """That an error is raised if feeding DatasetDicts made of Datasets that
    do not all have the same features.

    This is not strictly required, but featurising DatasetDicts of heterogeneous
    collections of Datasets is not part of any official workflow we would want to
    support. This makes sures we highlight this kind of inconsistencies as early
    as possible, preventing hard to debug side effects down the road.
    """
    ds_train = datasets.Dataset.from_dict({"a": [1, 2], "c": [3, 4]})
    ds_test = datasets.Dataset.from_dict({"a": [2, 1], "b": [4, 3]})
    dataset_dict = datasets.DatasetDict({"train": ds_train, "test": ds_test})

    representations = fixture_mock_representations
    column_to_featurise = dataset_dict.column_names["train"][0]

    with pytest.raises(ValueError, match=r".*Inconsistent input features.*"):
        featurise_dataset(
            dataset=dataset_dict,
            column=column_to_featurise,
            representations=representations,
        )
