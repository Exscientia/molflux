"""Uncertainty measure: Computes a given metric over a range of thresholded datasets based on uncertainty."""

import logging
from typing import Any, Optional

import evaluate
import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

import datasets
from molflux.metrics import load_metric
from molflux.metrics.bases import PredictionIntervalMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_CITATION = """\
@article{leibig2017leveraging,
  title={Leveraging uncertainty information from deep neural networks for disease detection},
  author={Leibig, Christian and Allken, Vaneeda and Ayhan, Murat Seckin and Berens, Philipp and Wahl, Siegfried},
  journal={Scientific reports},
  volume={7},
  number={1},
  pages={1--14},
  year={2017},
  publisher={Nature Publishing Group}
}
"""

_DESCRIPTION = """\
Computes regression metrics over a range of uncertainty thresholds.
Measures the uncertainty quality, eg. we expect performance to get worse when including predictions with higher
uncertainty.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated targets as returned by a classifier/regressor.
    references: Ground truth (correct) target values.
    uncertainties: Estimated uncertainties for the predictions. The lower the value the more confident that the
                prediction is correct.
    metric_name (optional): The name of the metric, used to compute performance over the alternating thresholded
        dataset. All metrics from the catalogue are possible. Defaults to mean_squared_error.
    num_of_threshold_steps: The number of thresholds (unique values in `uncertainties`) to use for computing curve
        (evenly sampled from the threshold distribution). Used to speed up computation. (default = -1, use all thresholds)

Returns:
    uncertainty_based_rejection: Nested dictionary. For each threshold (unique values in uncertainties), there is one
        dictionary with an integer as key and a result dictionary as value:
                metric_name --> computed metric as requested via metric_name
                uncertainty_thr --> All samples with <= uncertainty value were included in the analysis.
                size_of_thresholded_ds --> Number of samples in the thresholded dataset.
                frac_of_retained_data --> Fraction of the thresholded dataset in comparison to the original dataset.

        If num_of_theshold_steps != -1, the number of molfluxs of the dictionary will be less.

Examples:
    >>> from molflux.metrics import load_metric
    >>> m = load_metric("uncertainty_based_rejection")
    >>> ref = [0, 1, 2, 3, 4]
    >>> pred = [0, 2, 2, 3, 5]
    >>> uncertainties = [0, 1, 0.2, 0.2, 1.5]
    >>> result = m.compute(references=ref, predictions=pred, uncertainties=uncertainties)
    >>> result["uncertainty_based_rejection"][1]
    {'mean_squared_error': 0.0, 'uncertainty_thr': 0.2..., 'size_of_thresholded_ds': 3, 'frac_of_retained_data': 0.6}
    >>> result = m.compute(references=ref, predictions=pred, uncertainties=uncertainties, metric_name='max_error')
    >>> result["uncertainty_based_rejection"][2]
    {'max_error': 1.0, 'uncertainty_thr': 1.0, 'size_of_thresholded_ds': 4, 'frac_of_retained_data': 0.8}

"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class UncertaintyBasedRejection(PredictionIntervalMetric):
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
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        uncertainties: Optional[ArrayLike] = None,
        prediction_intervals: Optional[ArrayLike] = None,
        metric_name: str = "mean_squared_error",
        progress_bar: bool = True,
        num_of_threshold_steps: int = -1,
        override_huggingface: bool = False,
        **kwargs: Any,
    ) -> MetricResult:
        """

        Args:
            predictions:
            references:
            uncertainties:
            metric_name:
            progress_bar:
            num_of_threshold_steps: number of thresholds to use for computing curve (evenly sampled from the threshold
            distribution) (default = -1, use all thresholds)
            override_huggingface: override the hugging face compute method by using the _compute of the metrics.
            makes computation faster, but might result in unstable behaviour!!!
            **kwargs:

        Returns:

        """

        if uncertainties is not None:
            uncertainties = np.array(uncertainties)
        elif prediction_intervals is not None:
            lower_bound, upper_bound = zip(*prediction_intervals)
            uncertainties = np.array(upper_bound) - np.array(lower_bound)
        else:
            raise ValueError(
                "Please provide either uncertainties or prediction intervals.",
            )
        predictions = np.array(predictions)
        references = np.array(references)
        thresholds = list(np.unique(uncertainties))

        # if num_of_threshold_steps < len(thresholds), subsample thresholds by linear interpolation
        if (num_of_threshold_steps != -1) and (
            num_of_threshold_steps < len(thresholds)
        ):
            # interpolator
            threshold_interpolator = interp1d(
                np.linspace(0, 1, len(thresholds)),
                thresholds,
            )

            # subsample
            thresholds = threshold_interpolator(
                np.linspace(0, 1, num_of_threshold_steps),
            )

        exs_metric = load_metric(metric_name)

        results_dic = {}
        for ii, thr in enumerate(
            tqdm(
                thresholds,
                "Computing uncertainty based rejection",
                disable=(not progress_bar),
            ),
        ):
            mask = uncertainties <= thr
            pred_filtered = predictions[mask]
            ref_filtered = references[mask]
            if len(ref_filtered) > 1:
                if override_huggingface:
                    results = exs_metric._compute(  # type: ignore[attr-defined]
                        references=ref_filtered,
                        predictions=pred_filtered,
                    )
                else:
                    results = exs_metric.compute(
                        references=ref_filtered,
                        predictions=pred_filtered,
                    )

                results["uncertainty_thr"] = thr
                results["size_of_thresholded_ds"] = len(pred_filtered)
                results["frac_of_retained_data"] = len(pred_filtered) / float(
                    len(predictions),
                )
                results_dic[ii] = results

        return {self.tag: results_dic}
