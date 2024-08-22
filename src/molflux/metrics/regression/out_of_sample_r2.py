"""Out of sample R2"""

import logging
from typing import Any

import evaluate
from sklearn.metrics import mean_squared_error

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Out of sample R2.

Best possible score is 1.0 and it can be negative (because the model can be
arbitrarily worse). This metric overcomes some of the limitations of R2.
While R2 is comparing a perfect model and a constant model, when evaluated on
a test set it uses the test labels so in some sense has information leakage.
To overcome this the out of sample version of R2 uses the mean of training
labels rather than test labels. The mean of the training labels must be provided
as an additional argument, `dummy_predictions`. Note that this metric could
conceivably exaggerate the performance of a general model on a project
test set compared to a project model.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated target values.
    references: Ground truth (correct) target values.
    dummy_predictions: Estimated value from a dummy model constant on the training data.

Returns:
    oos_r2: The out of sample R2 score.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("out_of_sample_r2")
    >>> predictions = [2.5, 0.0, 2, 8]
    >>> references = [3, -0.5, 2, 7]
    >>> dummy_prediction = 3.125
    >>> metric.compute(predictions=predictions, references=references, dummy_prediction=dummy_prediction)
    {'out_of_sample_r2': 0.949...}
"""

_CITATION = """\
https://towardsdatascience.com/whats-wrong-with-r-squared-and-how-to-fix-it-7362c5f26c53
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class OutOfSampleR2(HFMetric):
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
                "https://towardsdatascience.com/whats-wrong-with-r-squared-and-how-to-fix-it-7362c5f26c53",
            ],
        )

    def _score(  # type: ignore[override]
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        dummy_prediction: float,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute out-of-sample R-squared."""
        mse_pred = mean_squared_error(references, predictions)
        mse_dummy = mean_squared_error(
            references,
            [dummy_prediction] * len(predictions),
        )
        mse_dummy = max(mse_dummy, 10**-16)
        oos_r2 = 1 - (mse_pred / mse_dummy)
        return {self.tag: oos_r2}
