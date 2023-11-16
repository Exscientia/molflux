"""Maximum residual error."""

import logging
from typing import Any

import evaluate
from sklearn.metrics import max_error

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
This metric computes the maximum residual error, a metric that captures the
worst case error between the predicted value and the true value. In a perfectly
fitted single output regression model, the maximum residual error would be 0 on
the training set and though this would be highly unlikely in the real world,
this metric shows the extent of error that the model had when it was fitted.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated target values.
    references: Ground truth (correct) target values.

Returns:
    max_error: A positive floating point value (the best value is 0.0).

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("max_error")
    >>> predictions = [9, 2, 7, 1]
    >>> references = [3, 2, 7, 1]
    >>> metric.compute(predictions=predictions, references=references)
    {'max_error': 6.0}
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
class MaxError(HFMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("float"),
                },
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        **kwargs: Any,
    ) -> MetricResult:
        score = max_error(y_true=references, y_pred=predictions)
        return {self.tag: score}
