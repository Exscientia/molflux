"""Computes an ROC curve over SMILES strings, i.e. "is the correct synthesis product in top-k predictions"."""

import logging
from typing import Any, List, Optional, Tuple

import evaluate
import numpy as np

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_CITATION = """\
@InProceedings{exscientia:accuracy_roc,
title = {Accuracy ROC},
authors={Exscientia},
year={2022}
}
"""

_DESCRIPTION = """\
Computes an accuracy ROC curve over SMILES strings, i.e. "is the correct synthesis product in top-k predictions".
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates what proportion of SMILES-encoded predictions hit the correct answer with the top k tries.

Args:
    predictions: list of lists of predictions to score. Each prediction should be a SMILES string.
    references: the correct answer for each set of predictions. Each
        reference should be a SMILES string.
    average: If `True`, an molflux-wise average across results is performed.

Returns:
    accuracy_roc: list whose k component is the answer to "what proportion
        of examples placed the correct answer within the first k predictions?"
        If 'average=True', "on average, what proportion of examples placed the correct answer within the first k predictions?"

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("top_k_accuracy_roc")
    >>> predictions = [['A', 'B', 'C', 'D', 'E']]
    >>> references = ['B']
    >>> metric.compute(predictions=predictions, references=references)
    {'top_k_accuracy_roc': [[0.0, 1.0, 1.0, 1.0, 1.0]]}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TopKAccuracyRoc(HFMetric):
    """Computes the accuracy ROC curve.

    Useful for synthesis prediction."""

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
        average: bool = False,
        **kwargs: Any,
    ) -> MetricResult:
        """Returns the accuracy ROC curve"""

        max_k = len(predictions[0])

        def get_pred_idx(preds: List[str], correct: str) -> int:
            # Return max_k if none of the predictions match
            return next((i for i, pred in enumerate(preds) if pred == correct), max_k)

        idxs = [
            get_pred_idx(preds, product)
            for preds, product in zip(predictions, references)
        ]

        # first for each idx in idxs define a vector that is 0 until idx (exclusive) and 1 after
        # note that max_k is larger than any i in range(max_k) so float(i >= max_k) = 0.
        present_at_idx = np.array(
            [[float(j >= idx) for j in range(max_k)] for idx in idxs],
        )

        if average:
            present_at_idx = present_at_idx.mean(axis=0, keepdims=True)

        accuracy_roc: List[float] = present_at_idx.tolist()

        return {self.tag: accuracy_roc}
