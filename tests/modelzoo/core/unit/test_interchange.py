import pandas as pd

import datasets
from molflux.modelzoo.interchange import hf_dataset_from_dataframe


def test_is_noop_on_huggingface_datasets():
    """That the interchange function is pass-through for huggingface Datasets."""
    dataset = datasets.Dataset.from_dict({"A": [1, 2, 3], "B": [4, 5, 6]})
    out = hf_dataset_from_dataframe(dataset)
    assert out == dataset


def test_can_interchange_pandas_df_with_nested_fields():
    """That can interchange a pandas DataFrame with array-like features."""
    dataframe = pd.DataFrame({"A": [1, 2], "B": [[1, 2, 3], [4, 5, 6]]})
    out = hf_dataset_from_dataframe(dataframe)
    assert isinstance(out, datasets.Dataset)
