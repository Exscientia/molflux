"""R2  (coefficient of determination) regression score function."""

import logging
from typing import Any, List, Optional

import evaluate
from sklearn.metrics import r2_score

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
R2 (coefficient of determination) regression score function.

Best possible score is 1.0 and it can be negative (because the model can be
arbitrarily worse). A constant model that always predicts the expected value of
y, disregarding the input features, would get a R2 score of 0.0."""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated target values.
    references: Ground truth (correct) target values.
    sample_weight (optional): Weighting of each sample.
    multioutput (optional): Defines aggregating of multiple output scores.
        Array-like value defines weights used to average errors. Alternatively,
        one of {'raw_values', 'uniform_average', 'variance_weighted'}. Defaults
        to 'uniform_average'.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.

Returns:
    r2: The R2 score or ndarray of scores if 'multioutput' is 'raw_values'.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("r2")
    >>> predictions = [2.5, 0.0, 2, 8]
    >>> references = [3, -0.5, 2, 7]
    >>> metric.compute(predictions=predictions, references=references)
    {'r2': 0.948...}
    >>> references = [1, 2, 3]
    >>> predictions = [1, 2, 3]
    >>> metric.compute(predictions=predictions, references=references)
    {'r2': 1.0}
    >>> references = [1, 2, 3]
    >>> predictions = [2, 2, 2]
    >>> metric.compute(predictions=predictions, references=references)
    {'r2': 0.0}
    >>> references = [1, 2, 3]
    >>> predictions = [3, 2, 1]
    >>> metric.compute(predictions=predictions, references=references)
    {'r2': -3.0}
    >>> metric = load_metric("r2", config_name="multioutput")
    >>> predictions = [[0, 2], [-1, 2], [8, -5]]
    >>> references = [[0.5, 1], [-1, 1], [7, -6]]
    >>> metric.compute(predictions=predictions, references=references, multioutput='variance_weighted')
    {'r2': 0.938...}
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
class R2(HFMetric):
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
        score = r2_score(
            y_true=references,
            y_pred=predictions,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )
        return {self.tag: score}
