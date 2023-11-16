"""Balanced accuracy metric."""

import logging
from typing import Any, List, Optional

import evaluate
from sklearn.metrics import balanced_accuracy_score

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
This metric computes the balanced accuracy, which avoids inflated
performance estimates on imbalanced datasets. It is the macro-average of
recall scores per class or, equivalently, raw accuracy where each sample is
weighted according to the inverse prevalence of its true class. Thus for
balanced datasets, the score is equal to accuracy.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated targets as returned by a classifier.
    references: Ground truth (correct) target values.
    adjusted (optional): When ``True``, the result is adjusted for chance,
            so that random performance would score 0, while keeping perfect
            performance at a score of 1. Defaults to ``False``.
    sample_weight (optional): Weighting of each sample.

Returns:
    balanced_accuracy: balanced accuracy score

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("balanced_accuracy")
    >>> references = [0, 1, 0, 0, 1, 0]
    >>> predictions = [0, 1, 0, 0, 0, 1]
    >>> metric.compute(references=references, predictions=predictions)
    {'balanced_accuracy': 0.625}
"""

_CITATION = """\
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BalancedAccuracy(HFMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                },
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        sample_weight: Optional[List[float]] = None,
        adjusted: bool = False,
        **kwargs: Any,
    ) -> MetricResult:
        score = balanced_accuracy_score(
            y_true=references,
            y_pred=predictions,
            adjusted=adjusted,
            sample_weight=sample_weight,
        )
        return {self.tag: float(score)}
