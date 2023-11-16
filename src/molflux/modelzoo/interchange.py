from typing import Any

import pandas
import pyarrow.interchange

import datasets


def hf_dataset_from_dataframe(df: Any) -> datasets.Dataset:
    """Builds a huggingface datasets.Dataset from any DataFrame supporting the
    DataFrame Interchange Protocol.

    Huggingface datasets do not support the interchange protocol quite yet.
    We therefore first interchange the input dataframe into a pyarrow table
    and then build a dataset from it.

    References:
        https://data-apis.org/dataframe-protocol/latest/purpose_and_scope.html
    """

    if isinstance(df, datasets.Dataset):
        return df

    # pandas DataFrames are handled separately as they are not always compatible
    # with the pyarrow.interchange call down below
    if isinstance(df, pandas.DataFrame):
        return datasets.Dataset.from_pandas(df)

    try:
        return datasets.Dataset(pyarrow.interchange.from_dataframe(df))
    except NotImplementedError as e:
        raise TypeError(
            f"Could not interchange input dataframe of type {type(df)!r} into an arrow-backed dataframe object. Most likely, one of the columnar data types is not supported by Apache Arrow (pyarrow) backends.",
        ) from e
