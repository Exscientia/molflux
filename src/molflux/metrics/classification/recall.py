"""Recall score."""

import logging
from typing import Any, List, Optional, Union

import evaluate
from sklearn.metrics import recall_score

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Compute the recall score.
The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
true positives and ``fn`` the number of false negatives. The recall is
intuitively the ability of the classifier to find all the positive samples.
The best value is 1 and the worst value is 0.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated targets as returned by a classifier.
    references: Ground truth (correct) target values.
    average (optional): {'micro', 'macro', 'samples','weighted', 'binary'} or
        None, default='binary'. This parameter is required for
        multiclass/multilabel targets. If ``None``, the scores for each class
        are returned. Otherwise, this determines the type of averaging performed
        on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    labels (optional): The set of labels to include when
        ``average != 'binary'``, and their order if ``average is None``.
        Labels present in the data can be excluded, for example to calculate a
        multiclass average ignoring a majority negative class, while labels not
        present in the data will result in 0 components in a macro average.
        For multilabel targets, labels are column indices. By default, all
        labels in ``y_true`` and ``y_pred`` are used in sorted order.
    pos_label (optional): The class to report if ``average='binary'`` and the
        data is binary. If the data are multiclass or multilabel, this will be
        ignored; setting ``labels=[pos_label]`` and ``average != 'binary'``
        will report scores for that label only. Defaults to 1.
    sample_weight (optional): Weighting of each sample.
    zero_division (optional): "warn", 0 or 1. Sets the value to return when
        there is a zero division, i.e. when all predictions and labels are
        negative. If set to "warn", this acts as 0, but warnings are also
        raised. Defaults to "warn".

Returns:
    recall: Recall of the positive class in binary classification or
    weighted average of the precision of each class for the multiclass task.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("recall")
    >>> references = [0, 1, 2, 0, 1, 2]
    >>> predictions = [0, 2, 1, 0, 0, 1]
    >>> metric.compute(references=references, predictions=predictions, average="macro")
    {'recall': 0.33...}
    >>> metric.compute(references=references, predictions=predictions, average="micro")
    {'recall': 0.33...}
    >>> metric.compute(references=references, predictions=predictions, average="weighted")
    {'recall': 0.33...}
    >>> metric.compute(references=references, predictions=predictions, average=None)
    {'recall': [1.0, 0.0, 0.0]}
    >>> references = [0, 0, 0, 0, 0, 0]
    >>> metric.compute(references=references, predictions=predictions, average=None, zero_division=0.0)
    {'recall': [0.5, 0.0, 0.0]}
    >>> metric.compute(references=references, predictions=predictions, average=None, zero_division=1.0)
    {'recall': [0.5, 1.0, 1.0]}

Notes:
    When ``true positive + false negative == 0``, recall returns 0 and raises
    ``UndefinedMetricWarning``. This behavior can be modified with
    ``zero_division``.
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
class Recall(HFMetric):
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
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        sample_weight: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
        pos_label: int = 1,
        average: Optional[str] = "binary",
        zero_division: Union[int, str] = "warn",
        **kwargs: Any,
    ) -> MetricResult:
        score = recall_score(
            y_true=references,
            y_pred=predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        return {self.tag: score.tolist()}
