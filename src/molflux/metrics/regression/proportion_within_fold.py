"""Proportion of estimates within a factor of n of the true value"""

import logging
from typing import Any

import evaluate
import numpy as np

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Assessment of the proportion of estimates that lie within a factor of n (n-fold) of the true value
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated target values.
    references: Ground truth (correct) target values.
    n_fold (optional): Float defining the factor within which predictions are assessed
        compared to references. Defaults to 3.0
    log_scaled_inputs (optional): Bool indicating whether inputs are already log-transformed. If false will
        perform log transform, but assumes inputs in the positive reals. Defaults to False

Returns:
    proportion_within_fold: proportion of predictions that lie within n-fold region

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("proportion_within_fold")
    >>> predictions = [2.5, 4.0, 2, 8]
    >>> references = [3, 0.5, 2, 7]
    >>> metric.compute(predictions=predictions, references=references)
    {'proportion_within_fold': 0.75}
"""

_CITATION = """\
@article{ring2011phrma,
  title={PhRMA CPCDC initiative on predictive models of human pharmacokinetics, part 3: comparative assessement of prediction methods of human clearance},
  author={Ring, Barbara J and Chien, Jenny Y and Adkison, Kimberly K and Jones, Hannah M and Rowland, Malcolm and Jones, Rhys Do and Yates, James WT and Ku, M Sherry and Gibson, Christopher R and He, Handan and others},
  journal={Journal of pharmaceutical sciences},
  volume={100},
  number={10},
  pages={4090--4110},
  year={2011},
  publisher={Wiley Online Library}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ProportionWithinFold(HFMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float64"),
                    "references": datasets.Value("float64"),
                },
            ),
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        n_fold: float = 3.0,
        log_scaled_inputs: bool = False,
        **kwargs: Any,
    ) -> MetricResult:
        pred = np.array(predictions)
        ref = np.array(references)
        if not log_scaled_inputs:
            if np.any(pred <= 0) or np.any(ref <= 0):
                raise RuntimeError(
                    "Proportion within fold used with negative predictions or references, but may not make sense",
                )
            pred = np.log10(pred)
            ref = np.log10(ref)
        abs_errors = np.abs(pred - ref)
        score = sum(abs_errors <= np.log10(n_fold)) / ref.shape[0]
        return {self.tag: score}
