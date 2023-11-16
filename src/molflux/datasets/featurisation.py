import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
from more_itertools import zip_broadcast

import datasets
from molflux.datasets.interfaces import Representation, Representations
from molflux.datasets.typing import DatasetType, DisplayNames

_DEFAULT_DISPLAY_NAMES_TEMPLATE = "{source_column}::{feature_name}"

FreeformDisplayNames = Union[
    Optional[str],
    List[Union[Optional[str], List[Optional[str]]]],
]


logger = logging.getLogger(__name__)


def _to_canonical_display_names(
    display_names: FreeformDisplayNames,
    representations: Representations,
) -> DisplayNames:
    """Parses free-form display names provided by users into their canonical form.

    Display names in canonical form are represented as a nested list of display
    names, with one (list) entry for each representation used for featurisation.
    For example, custom display names for a collection of three representations
    of which the second generates two features, would be represented as follows:
    [["a"], ["b0", "b1"], ["c"]]

    Display names provided by users should ideally already be in canonical form,
    but alternatives are accepted for convenience. This function parses them into
    canonical form:

    - a non-nested list: this can be used as a shorthand by users using a single
        representation (instead of having to create a single-molflux nested list)
    - a single molflux: a.k.a. a 'template' string. This can be used to provide
        a template that will be applied to all output names across all
        representations. If 'None', a default fallback template will be applied.

    At any level, template strings (or None) can be used to broadcast the
    template across all associated outputs. For example, the following canonical
    form display names will result in the same output display names:
        [["a"], [None, None], ["c"]]; [["a"], [None], ["c"]]
    The first example explicitly applies the default template to each output
    feature of the second representation, whereby the second example implicitly
    broadcasts the default template to all outputs (in case the number is not
    known in advance)

    Returns:
        The canonical form display names for downstream processing.
    """

    explanation_banner = """\n
    Display names should be provided as a nested list of display
    names, with one (list) entry for each representation used for featurisation.
    For example, custom display names for a collection of three representations
    of which the second generates two features, would be represented as follows:
    [["a"], ["b0", "b1"], ["c"]]
    """

    # avoids having to force 'representations' protocol to be Sized
    num_representations = len(list(representations))

    # 1) string template (or None): str_template -> [[str_template], ...]
    if display_names is None or isinstance(display_names, str):
        canonical_form = [[display_names] for _ in range(num_representations)]

    # 2) convenience converter for single representation notation: [x y ...] -> [[x y z]]
    elif isinstance(display_names, list) and all(
        x is None or isinstance(x, str) for x in display_names
    ):
        if num_representations != 1:
            raise ValueError(
                f"The provided display names syntax can only be used for single representations: {display_names!r} (you are using {num_representations})"
                + explanation_banner,
            )
        canonical_form = [display_names]  # type: ignore[list-item]

    # 3) inputs already in canonical form:
    elif isinstance(display_names, list) and all(
        isinstance(x, list) for x in display_names
    ):
        canonical_form = display_names  # type: ignore[assignment]

    # Reject unsupported free-form input syntax
    else:
        raise ValueError(
            f"The provided display names do not follow the accepted syntax: {display_names!r} (for {num_representations} representation(s))"
            + explanation_banner,
        )

    # Assert outputs are in canonical form (internal railguard)
    if (
        not all(isinstance(x, list) for x in canonical_form)
        or not len(canonical_form) == num_representations
    ):
        raise RuntimeError(
            f"Error parsing display names: {display_names!r} >> {canonical_form!r} (for {num_representations} representation(s))",
        )

    return canonical_form


def _consolidate_map_outputs(dataset: DatasetType) -> DatasetType:
    """Consolidates datasets returned by a `datasets.map` operation.

    One potential problem is that internally empty datasets are not processed
    by the `datasets.map` call. This can lead to inconsistencies in output
    feature names across empty and non-empty datasets.
    """
    if isinstance(dataset, datasets.Dataset):
        return dataset

    empty_splits = [k for k, v in dataset.num_rows.items() if v == 0]
    if not empty_splits:
        return dataset

    non_empty_splits = [k for k, v in dataset.num_rows.items() if v != 0]
    if not non_empty_splits:
        logger.warning(
            "Could not consolidate DatasetDict output features: all Datasets in the dict are empty",
        )
        return dataset

    # Replace all empty splits with empty datasets matching non-empty datasets
    reference = non_empty_splits[0]
    for k in empty_splits:
        empty_clone = datasets.Dataset(dataset[reference].data.slice(0, 0))
        dataset[k] = empty_clone

    return dataset


def featurise_dataset(
    dataset: DatasetType,
    column: str,
    representations: Union[Representation, Representations],
    display_names: FreeformDisplayNames = None,
    **map_kwargs: Any,
) -> DatasetType:
    """Featurises a dataset column according to the given representations.

    Args:
        dataset: The dataset to featurise.
        column: The name of the dataset column to featurise.
        representations: The representation or representations to featurise the column with.
        display_names: A list of custom labels to assign to the newly
            featurised columns, or a single string template that will be
            used to dynamically generate labels.
        map_kwargs: Optional keyword arguments to be passed to the underlying
            dataset's .map() method during featurisation.

    Returns:
        The featurised dataset.
    """

    # Make sure we can always iterate over a collection
    if not isinstance(representations, Representations):
        representations = [representations]

    # Make sure that all Datasets in the DatasetDict have the same features
    if isinstance(dataset, datasets.DatasetDict):
        all_features_match = len(set(map(tuple, dataset.column_names.values()))) == 1
        if not all_features_match:
            raise ValueError(
                f"Inconsistent input features across splits: got {dataset.column_names!r}",
            )

    canonical_display_names = _to_canonical_display_names(
        display_names,
        representations=representations,
    )

    # Apply batched featurisation instead of row-by-row
    if "batched" not in map_kwargs:
        map_kwargs["batched"] = True

    try:
        featurized_dataset = dataset.map(
            function=_featurise_batch,
            fn_kwargs={
                "column": column,
                "representations": representations,
                "display_names": canonical_display_names,
            },
            **map_kwargs,
        )
    except pa.ArrowInvalid as e:
        raise TypeError(
            "Apache Arrow serialisation error: one or more representations might be incompatible with Apache Arrow backends.",
        ) from e

    featurized_dataset = _consolidate_map_outputs(featurized_dataset)

    return featurized_dataset


def _featurise_batch(
    example: Dict[str, Any],
    column: str,
    representations: Representations,
    display_names: DisplayNames,
) -> Dict[str, Any]:
    """Featurises a batch's column according to the given representations.

    The primary objective of batch mapping is to speed up processing. Often
    times, it is faster to work with batches of data instead of single examples.
    """

    if column not in example:
        raise KeyError(
            f"Feature {column!r} not in dataset: available features are {list(example.keys())!r}",
        )

    samples = example[column]

    for representation, representation_display_names in zip(
        representations,
        display_names,
    ):
        # this could be a multi-output feature
        representation_results = representation.featurise(samples)

        # handle template strings as display names for one-to-many representations
        if len(representation_display_names) == 1 and len(representation_results) != 1:
            display_name = representation_display_names[0]
            is_template = display_name is None or (
                "{" in display_name and "}" in display_name
            )
            if not is_template:
                raise ValueError(
                    f"Only template display names can be used as placeholders to be broadcasted: {display_name!r}",
                )
            representation_display_names = display_name  # type: ignore[assignment]

        # templates should be broadcasted to apply to all columns of the variably sized output
        for (feature_name, result), display_name in zip_broadcast(  # type: ignore[misc]
            representation_results.items(),  # type: ignore[arg-type]
            representation_display_names,  # type: ignore[arg-type]
        ):
            ctx = {"source_column": column, "feature_name": feature_name}  # type: ignore[has-type]
            featurised_column_name = _template(
                display_name,
                default_template=_DEFAULT_DISPLAY_NAMES_TEMPLATE,
                **ctx,
            )

            if featurised_column_name in example:
                warnings.warn(
                    f"An existing column is being overwritten on featurisation: {column}::{feature_name} >> {featurised_column_name}",  # type: ignore[has-type]
                    stacklevel=1,
                )

            example[featurised_column_name] = result  # type: ignore[has-type]

    return example


def _template(target: Optional[str], default_template: str, **ctx: Any) -> str:
    """Templates a string with local context information.

    Args:
        target: The string to template. If None, the default template is used.
        default_template: The default template to use.
        ctx: Optional context that will be used to template the string.

    Returns:
        The templated string.
    """
    if target is None:
        target = default_template

    try:
        templated = target.format(**ctx)
    except KeyError as e:
        raise KeyError(
            f"Could not template the target string: {e} missng from context: {ctx!r}.",
        ) from e

    return templated
