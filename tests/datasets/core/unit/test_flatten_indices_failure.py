import datasets


def test_flatten_indices():
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

    # step to filter indices with because of Batched=True
    dataset_filtered.flatten_indices()

    assert True
