"""Median absolute error regression loss."""

import logging
from typing import Any, List, Optional

import evaluate
from sklearn.metrics import median_absolute_error

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
The median_absolute_error is particularly interesting because it is robust to
outliers. The loss is calculated by taking the median of all absolute
differences between the target and the prediction.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated target values.
    references: Ground truth (correct) target values.
    sample_weight (optional): Weighting of each sample.
    multioutput (optional): Defines aggregating of multiple output scores.
        Array-like value defines weights used to average errors. Alternatively,
        one of {'raw_values', 'uniform_average'}. Defaults
        to 'uniform_average'.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

Returns:
    median_absolute_error: If multioutput is `raw_values`, then mean absolute
    error is returned for each output separately. If multioutput is
    `uniform_average` or an ndarray of weights, then the weighted average of
    all output errors is returned.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("median_absolute_error")
    >>> predictions = [2.5, 0.0, 2, 8]
    >>> references = [3, -0.5, 2, 7]
    >>> metric.compute(predictions=predictions, references=references)
    {'median_absolute_error': 0.5}
    >>> metric = load_metric("median_absolute_error", config_name="multioutput")
    >>> predictions = [[0, 2], [-1, 2], [8, -5]]
    >>> references = [[0.5, 1], [-1, 1], [7, -6]]
    >>> metric.compute(predictions=predictions, references=references)
    {'median_absolute_error': 0.75}
    >>> metric.compute(predictions=predictions, references=references, multioutput='raw_values')
    {'median_absolute_error': [0.5, 1.0]}
    >>> metric.compute(predictions=predictions, references=references, multioutput=[0.3, 0.7])
    {'median_absolute_error': 0.85}

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
class MedianAbsoluteError(HFMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Sequence(datasets.Value("float")),
                }
                if self.config_name == "multioutput"
                else {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("float"),
                },
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        sample_weight: Optional[List[float]] = None,
        multioutput: str = "uniform_average",
        **kwargs: Any,
    ) -> MetricResult:
        score = median_absolute_error(
            y_true=references,
            y_pred=predictions,
            sample_weight=sample_weight,
            multioutput=multioutput,
        ).tolist()
        return {self.tag: score}
