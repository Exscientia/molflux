"""Computes a diversity ROC curve over SMILES string."""

import logging
from typing import Any, List, Optional, Tuple

import evaluate
import numpy as np

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_CITATION = """\
@InProceedings{exscientia:diversity_roc,
title = {Diversity ROC},
authors={Exscientia},
year={2022}
}
"""

_DESCRIPTION = """\
Computes a diversity ROC curve over SMILES strings, i.e. "how many SMILES in the first k results are diverse"?
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates what proportion of SMILES-encoded predictions are diverse within the top k tries.

Args:
    predictions: list of lists of predictions to score. Each prediction should be a SMILES string.
    references:  list of references. Ignored by this metric, but should be of the same length as 'predictions'.
    average: If `True`, an molflux-wise average across results is performed.

Returns:
    diversity_roc: list whose k component is the answer to "how many diverse strings are in top-k predictions".
        If 'average=True', "on average, how many diverse strings are in top-k predictions".

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("diversity_roc")
    >>> predictions = [["CC", "C", "CC", "C", "CCC"]]
    >>> references = [None]
    >>> metric.compute(predictions=predictions, references=references)
    {'diversity_roc': [[1.0, 1.0, 0.66..., 0.5, 0.6]]}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class DiversityRoc(HFMetric):
    """Computes the diversity ROC curve."""

    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction
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
        average: bool = False,
        **kwargs: Any,
    ) -> MetricResult:
        """Returns the diversity ROC curve"""

        max_k = len(predictions[0])

        # vector of 0./1. representing whether a prediction is repeated
        def mark_new_predictions(preds: List[str]) -> np.ndarray:
            idxs = sorted([preds.index(p) for p in set(preds)])
            new_predictions = np.zeros((max_k,))
            new_predictions[idxs] = 1
            return new_predictions

        new_predictions = [mark_new_predictions(preds) for preds in predictions]

        def running_mean(x: np.ndarray) -> np.ndarray:
            cumsum = np.cumsum(x)
            norm = np.arange(1, np.shape(x)[0] + 1)
            mean: np.ndarray = cumsum / norm
            return mean

        running_diversities = np.array([running_mean(p) for p in new_predictions])

        if average:
            running_diversities = running_diversities.mean(axis=0, keepdims=True)

        diversity_roc: List[float] = running_diversities.tolist()

        return {self.tag: diversity_roc}
