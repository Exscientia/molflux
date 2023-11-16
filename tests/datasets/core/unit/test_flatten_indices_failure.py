import pytest

import datasets


@pytest.mark.xfail()
def test_flatten_indices_failure():
    # dataset with array and nested array with defined length
    dataset = datasets.Dataset.from_dict(
        {
            "array_nested": [[[0] * 3] * 7] * 2000,
            "array": [[0] * 3] * 2000,
        },
        features=datasets.Features(
            {
                "array_nested": datasets.Sequence(
                    datasets.Sequence(
                        datasets.Value(dtype="int64", id=None),
                        length=3,
                        id=None,
                    ),
                    length=-1,
                    id=None,
                ),
                "array": datasets.Sequence(
                    datasets.Value(dtype="int64", id=None),
                    length=3,
                    id=None,
                ),
            },
        ),
    )

    # a filter operation which does nothing (checks that all nested arrays have len < 8, they have len == 7)
    dataset_filtered = dataset.filter(lambda x: len(x["array"]) < 8)

    # step that fails because of Batched=True in map
    dataset_filtered.flatten_indices()


def test_flatten_indices_failure_workaround_1():
    # dataset with array and nested array with defined length
    dataset = datasets.Dataset.from_dict(
        {
            "array_nested": [[[0] * 3] * 7] * 2000,
            "array": [[0] * 3] * 2000,
        },
        features=datasets.Features(
            {
                "array_nested": datasets.Sequence(
                    datasets.Sequence(
                        datasets.Value(dtype="int64", id=None),
                        length=3,
                        id=None,
                    ),
                    length=-1,
                    id=None,
                ),
                "array": datasets.Sequence(
                    datasets.Value(dtype="int64", id=None),
                    length=3,
                    id=None,
                ),
            },
        ),
    )

    # a filter operation which does nothing (checks that all nested arrays have len < 8, they have len == 7)
    dataset_filtered = dataset.filter(lambda x: len(x["array"]) < 8)

    # do a map with Batched=False by default
    dataset_filtered = dataset_filtered.map()

    # This now runs fine!
    dataset_filtered.flatten_indices()

    assert True


def test_flatten_indices_failure_workaround_2():
    # dataset with array and nested array with defined length
    dataset = datasets.Dataset.from_dict(
        {
            "array_nested": [[[0] * 3] * 7] * 2000,
            "array": [[0] * 3] * 2000,
        },
        features=datasets.Features(
            {
                "array_nested": datasets.Sequence(
                    datasets.Sequence(
                        datasets.Value(dtype="int64", id=None),
                        length=3,
                        id=None,
                    ),
                    length=-1,
                    id=None,
                ),
                "array": datasets.Sequence(
                    datasets.Value(dtype="int64", id=None),
                    length=3,
                    id=None,
                ),
            },
        ),
    )

    # a filter operation which does nothing (checks that all nested arrays have len < 8, they have len == 7)
    dataset_filtered = dataset.filter(lambda x: len(x["array"]) < 8)

    # cast to features with no defined length
    dataset_filtered = dataset_filtered.cast(
        datasets.Features(
            {
                "array_nested": datasets.Sequence(
                    datasets.Sequence(
                        datasets.Value(dtype="int64", id=None),
                        length=-1,
                        id=None,
                    ),
                    length=-1,
                    id=None,
                ),
                "array": datasets.Sequence(
                    datasets.Value(dtype="int64", id=None),
                    length=-1,
                    id=None,
                ),
            },
        ),
    )

    # This now runs fine!
    dataset_filtered.flatten_indices()

    assert True
