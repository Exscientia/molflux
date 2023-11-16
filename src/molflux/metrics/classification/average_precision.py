"""Compute average precision (AP) from prediction scores."""

import logging
from typing import Any, List, Optional, Union

import evaluate
from sklearn.metrics import average_precision_score

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold,
with the increase in recall from the previous threshold used as the weight. This implementation is not
interpolated and is different from computing the area under the precision-recall curve with the trapezoidal
rule, which uses linear interpolation and can be too optimistic.

Note: this implementation is restricted to the binary classification task or multilabel classification task.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Target prediction scores.
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions.

    references:
        True binary labels or binary label indicators.

    average: {`micro`, `samples`, `weighted`, `macro`} or None, default=`macro`
        If None, the scores for each class are returned. Otherwise, this determines the type
        of averaging performed on the data:

        'micro':
            Calculate metrics globally by considering each molflux of the label indicator matrix as a label.

        'macro':
            Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance
            into account.

        'weighted':
            Calculate metrics for each label, and find their average, weighted by support (the number of true
            instances for each label).

        'samples':
            Calculate metrics for each instance, and find their average.

        Will be ignored when y_true is binary.

    pos_label: int or str, default=1
        The label of the positive class. Only applied to binary y_true. For multilabel-indicator y_true,
        pos_label is fixed to 1.

    sample_weight: array-like of shape (n_samples,), default=None
        Sample weights.

Returns:
    average_precision: The average precision score.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("average_precision")
    >>> references = [1, 0, 1, 1, 0, 0]
    >>> predictions = [0.5, 0.2, 0.99, 0.3, 0.1, 0.7]
    >>> metric.compute(references=references, predictions=predictions)
    {'average_precision': 0.80...}
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("average_precision", config_name="multilabel")
    >>> references = [[1, 1, 0],
    ...         [1, 1, 0],
    ...         [0, 1, 0],
    ...         [0, 0, 1],
    ...         [0, 1, 1],
    ...         [1, 0, 1]]
    >>> predictions = [[0.3, 0.5, 0.2],
    ...                 [0.7, 0.2, 0.1],
    ...                 [0.005, 0.99, 0.005],
    ...                 [0.2, 0.3, 0.5],
    ...                 [0.1, 0.1, 0.8],
    ...                 [0.1, 0.7, 0.2]]
    >>> results = metric.compute(references=references, predictions=predictions, average=None)
    >>> print([round(res, 2) for res in results['average_precision']])
    [0.87, 0.73, 0.92]
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
class AveragePrecision(HFMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("int32"),
                },
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        sample_weight: Optional[List[float]] = None,
        average: str = "macro",
        pos_label: Union[int, str] = 1,
        **kwargs: Any,
    ) -> MetricResult:
        score = average_precision_score(
            y_true=references,
            y_score=predictions,
            average=average,
            sample_weight=sample_weight,
            pos_label=pos_label,
        )
        return {self.tag: score}
