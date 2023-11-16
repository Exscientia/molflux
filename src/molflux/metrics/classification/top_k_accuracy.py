"""Top-k accuracy classification score."""

import logging
from typing import Any, List, Optional

import evaluate
from sklearn.metrics import top_k_accuracy_score

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
This metric computes the number of times where the correct label is among the
top k labels predicted (ranked by predicted scores). Note that the multilabel
case isn`t covered here.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Target scores. These can be either probability estimates or
        non-thresholded decision values. The binary case expects scores with
        shape (n_samples,) while the multiclass case expects scores with shape
        (n_samples, n_classes). In the multiclass case, the order of the class
        scores must correspond to the order of ``labels``, if provided, or else
        to the numerical or lexicographical order of the labels in
        ``references``. If ``references`` does not contain all the labels,
        ``labels`` must be provided.
    references: Ground truth (correct) target values.
    k (optional): Number of most likely outcomes considered to find the correct
        label. Defaults to 2.
    normalize (optional): If False, return the number of correctly classified
        samples. Otherwise, return the fraction of correctly classified samples.
    sample_weight (optional): Sample weights.
    labels (optional): Multiclass only. List of labels that index the classes
        in ``predictions``. If ``None``, the numerical or lexicographical order
        of  the labels in ``references`` is used. If ``predictions`` does not
        contain all the labels, ``labels`` must be provided.

Returns:
    top_k_accuracy: The top-k accuracy score. The best performance is 1 with
    `normalize == True` and the number of samples with `normalize == False`.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("top_k_accuracy", config_name="multiclass")
    >>> references = [0, 1, 2, 2]
    >>> predictions = [[0.5, 0.2, 0.2],  # 0 is in top 2
    ...                [0.3, 0.4, 0.2],  # 1 is in top 2
    ...                [0.2, 0.4, 0.3],  # 2 is in top 2
    ...                [0.7, 0.2, 0.1]] # 2 isn't in top 2
    >>> metric.compute(references=references, predictions=predictions, k=2)
    {'top_k_accuracy': 0.75}
    >>> # Not normalizing gives the number of "correctly" classified samples
    >>> metric.compute(references=references, predictions=predictions, k=2, normalize=False)
    {'top_k_accuracy': 3}
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
class TopKAccuracy(HFMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Value("int32"),
                },
            ),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.top_k_accuracy_score.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        sample_weight: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
        k: int = 2,
        normalize: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        score = top_k_accuracy_score(
            y_true=references,
            y_score=predictions,
            k=k,
            normalize=normalize,
            sample_weight=sample_weight,
            labels=labels,
        )
        return {self.tag: score}
