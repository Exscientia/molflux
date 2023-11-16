"""Computes an ROC curve over SMILES strings"""

import logging
from typing import Any, List, Optional, Tuple

import evaluate
import numpy as np

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_CITATION = """\
@InProceedings{exscientia:validity_roc,
title = {Validity ROC},
authors={Exscientia},
year={2022}
}
"""

_DESCRIPTION = """\
Computes a validity ROC curve over SMILES strings, i.e. "how many SMILES in the first k results are valid"?
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates what proportion of SMILES-encoded predictions are valid within the top k tries.

Args:
    predictions: list of lists of predictions to score. Each prediction should be a SMILES string.
    references:  list of references. Ignored by this metric, but should be of the same length as 'predictions'.
    invalid_string: the invalid string (default "<INVALID>").
    average: If `True`, an molflux-wise average across results is performed.

Returns:
    validity_roc: list whose k component is the answer to "is the correct synthesis product in top-k predictions"
        If 'average=True', "on average, is the correct synthesis product in top-k predictions"

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("validity_roc")
    >>> predictions = [["CC", "CC", None, None, "CCC"]]
    >>> references = [None]
    >>> metric.compute(predictions=predictions, references=references)
    {'validity_roc': [[1.0, 1.0, 0.66..., 0.5, 0.6]]}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ValidityRoc(HFMetric):
    """Computes the validity ROC curve."""

    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string")),
                    "references": datasets.Value("string"),
                },
            ),
        )

    def _pre_process_inputs(
        self,
        predictions: Optional[ArrayLike] = None,
        references: Optional[ArrayLike] = None,
    ) -> Tuple[Optional[ArrayLike], ...]:
        """Converts None inputs to strings to allow using None as invalid token.

        Otherwise, None molfluxs would raise as they do not count as string features.
        """
        if predictions is not None:
            predictions = [
                [p if p is not None else "None" for p in ps] for ps in predictions
            ]

        if references is not None:
            references = [r if r is not None else "None" for r in references]

        return predictions, references

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        invalid_token: Optional[str] = None,
        average: bool = False,
        **kwargs: Any,
    ) -> MetricResult:
        """Returns the validity ROC curve"""

        if invalid_token is None:
            invalid_token = "None"  # noqa: S105

        # vector of 0./1. representing (in)valid SMILES
        valid_preds = [
            [p != invalid_token for p in one_set_of_preds]
            for one_set_of_preds in predictions
        ]

        def running_mean(x: List[Any]) -> np.ndarray:
            cumsum = np.cumsum(x)
            norm = np.arange(1, np.shape(x)[0] + 1)
            mean: np.ndarray = cumsum / norm
            return mean

        running_means = np.array([running_mean(p) for p in valid_preds])

        if average:
            running_means = running_means.mean(axis=0, keepdims=True)

        validity_roc: List[float] = running_means.tolist()

        return {self.tag: validity_roc}
