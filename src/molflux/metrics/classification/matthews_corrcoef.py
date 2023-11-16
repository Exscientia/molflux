"""Matthews correlation coefficient (MCC)."""

import logging
from typing import Any, List, Optional

import evaluate
from sklearn.metrics import matthews_corrcoef

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
The Matthews correlation coefficient is used in machine learning as a measure of
the quality of binary and multiclass classifications. It takes into account true
and false positives and negatives and is generally regarded as a balanced
measure which can be used even if the classes are of very different sizes.
The MCC is in essence a correlation coefficient value between -1 and +1. A
coefficient of +1 represents a perfect prediction, 0 an average random
prediction and -1 an inverse prediction. The statistic is also known as the phi
coefficient.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated targets as returned by a classifier.
    references: Ground truth (correct) target values.
    sample_weight (optional): Weighting of each sample.

Returns:
    matthews_corrcoef: The Matthews correlation coefficient (+1 represents a
    perfect prediction, 0 an average random prediction and -1 and inverse
    prediction).

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("matthews_corrcoef")
    >>> references = [+1, +1, +1, -1]
    >>> predictions = [+1, -1, +1, +1]
    >>> metric.compute(references=references, predictions=predictions)
    {'matthews_corrcoef': -0.33...}
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
class MatthewsCorrcoef(HFMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                },
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        sample_weight: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        score = matthews_corrcoef(
            y_true=references,
            y_pred=predictions,
            sample_weight=sample_weight,
        )
        return {self.tag: float(score)}
