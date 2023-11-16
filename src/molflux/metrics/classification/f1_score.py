"""F1 score, also known as balanced F-score or F-measure"""

import logging
from typing import Any, List, Optional, Union

import evaluate
from sklearn.metrics import f1_score

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Compute the F1 score, also known as balanced F-score or F-measure.
The F1 score can be interpreted as a harmonic mean of the precision and
recall, where an F1 score reaches its best value at 1 and worst score at 0.
The relative contribution of precision and recall to the F1 score are
equal. The formula for the F1 score is::

    F1 = 2 * (precision * recall) / (precision + recall)

In the multi-class and multi-label case, this is the average of
the F1 score of each class with weighting depending on the ``average``
parameter.
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
    f1_score: F1 score of the positive class in binary classification or
    weighted average of the F1 scores of each class for the multiclass task.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("f1_score")
    >>> references = [0, 1, 2, 0, 1, 2]
    >>> predictions = [0, 2, 1, 0, 0, 1]
    >>> metric.compute(references=references, predictions=predictions, average="macro")
    {'f1_score': 0.26...}
    >>> metric.compute(references=references, predictions=predictions, average="micro")
    {'f1_score': 0.33...}
    >>> metric.compute(references=references, predictions=predictions, average="weighted")
    {'f1_score': 0.26...}
    >>> metric.compute(references=references, predictions=predictions, average=None)
    {'f1_score': [0.8, 0.0, 0.0]}
    >>> references = [0, 0, 0, 0, 0, 0]
    >>> predictions = [0, 0, 0, 0, 0, 0]
    >>> metric.compute(references=references, predictions=predictions, zero_division=1)
    {'f1_score': 1.0}

Notes:
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.
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
class F1Score(HFMetric):
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
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html",
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
        score = f1_score(
            y_true=references,
            y_pred=predictions,
            labels=labels,
            pos_label=pos_label,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        return {self.tag: score.tolist()}
