"""Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores."""

import logging
from typing import Any, List, Optional

import evaluate
from sklearn.metrics import roc_auc_score

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
from prediction scores.

The return values represent how well the model used is predicting the correct classes, based on the input data. A score of `0.5` means that the model is predicting exactly at chance, i.e. the model's predictions are correct at the same rate as if the predictions were being decided by the flip of a fair coin or the roll of a fair die. A score above `0.5` indicates that the model is doing better than chance, while a score below `0.5` indicates that the model is doing worse than chance.


Note: this implementation can be used with binary, multiclass and
multilabel classification, but some restrictions apply (see Args):
    - binary: The case in which there are only two different label classes, and each example gets only one label. This is the default implementation.
    - multiclass: The case in which there can be more than two different label classes, but each example still gets only one label.
    - multilabel: The case in which there can be more than two different label classes, and each example can have more than one label.

"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Target prediction scores.

        * In the binary case, it corresponds to an array of shape
          `(n_samples,)`. Both probability estimates and non-thresholded
          decision values can be provided. The probability estimates correspond
          to the **probability of the class with the greater label**,
          i.e. `estimator.classes_[1]` and thus
          `estimator.predict_proba(X, y)[:, 1]`. The decision values
          corresponds to the output of `estimator.decision_function(X, y)`.
          See more information in the :ref:`User guide <roc_auc_binary>`;
        * In the multiclass case, it corresponds to an array of shape
          `(n_samples, n_classes)` of probability estimates provided by the
          `predict_proba` method. The probability estimates **must**
          sum to 1 across the possible classes. In addition, the order of the
          class scores must correspond to the order of ``labels``,
          if provided, or else to the numerical or lexicographical order of
          the labels in ``y_true``. See more information in the
          :ref:`User guide <roc_auc_multiclass>`;
        * In the multilabel case, it corresponds to an array of shape
          `(n_samples, n_classes)`. Probability estimates are provided by the
          `predict_proba` method and the non-thresholded decision values by
          the `decision_function` method. The probability estimates correspond
          to the **probability of the class with the greater label for each
          output** of the classifier. See more information in the
          :ref:`User guide <roc_auc_multilabel>`.

    references: True labels or binary label indicators. The binary and
        multiclass cases expect labels with shape (n_samples,) while the
        multilabel case expects binary label indicators with shape
        (n_samples, n_classes).

    average (optional): {'micro', 'macro', 'samples','weighted'} or None,
        default='macro'.
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:
        Note: multiclass ROC AUC currently only handles the 'macro' and
        'weighted' averages.

        ``'micro'``:
            Calculate metrics globally by considering each molflux of the label
            indicator matrix as a label.

        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.

        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).

        ``'samples'``:
            Calculate metrics for each instance, and find their average.

        Will be ignored when ``y_true`` is binary.

    sample_weight (optional): Weighting of each sample.

    max_fpr (optional): If not ``None``, the standardized partial AUC [2]_ over
        the range [0, max_fpr] is returned. For the multiclass case,
        ``max_fpr``, should be either equal to ``None`` or ``1.0`` as AUC ROC
        partial computation currently is not supported for multiclass.

    multi_class (optional): {'raise', 'ovr', 'ovo'}, default='raise'.
        Only used for multiclass targets. Determines the type of configuration
        to use. The default value raises an error, so either
        ``'ovr'`` or ``'ovo'`` must be passed explicitly.

        ``'ovr'``:
            Stands for One-vs-rest. Computes the AUC of each class
            against the rest [3]_ [4]_. This
            treats the multiclass case in the same way as the multilabel case.
            Sensitive to class imbalance even when ``average == 'macro'``,
            because class imbalance affects the composition of each of the
            'rest' groupings.
        ``'ovo'``:
            Stands for One-vs-one. Computes the average AUC of all
            possible pairwise combinations of classes [5]_.
            Insensitive to class imbalance when
            ``average == 'macro'``.

    labels (optional): Only used for multiclass targets. List of labels that
        index the classes in ``predictions``. If ``None``, the numerical or
        lexicographical order of the labels in ``references`` is used. Defaults
        to ``None``.

Returns:
    roc_auc: Area Under the Receiver Operating Characteristic Curve.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("roc_auc")
    >>> references = [1, 0, 1, 1, 0, 0]
    >>> predictions = [0.5, 0.2, 0.99, 0.3, 0.1, 0.7]
    >>> results = metric.compute(references=references, predictions=predictions)
    >>> print(round(results['roc_auc'], 2))
    0.78
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("roc_auc", config_name="multiclass")
    >>> references = [1, 0, 1, 2, 2, 0]
    >>> predictions = [[0.3, 0.5, 0.2],
    ...                 [0.7, 0.2, 0.1],
    ...                 [0.005, 0.99, 0.005],
    ...                 [0.2, 0.3, 0.5],
    ...                 [0.1, 0.1, 0.8],
    ...                 [0.1, 0.7, 0.2]]
    >>> results = metric.compute(references=references, predictions=predictions, multi_class='ovr')
    >>> print(round(results['roc_auc'], 2))
    0.85
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("roc_auc", config_name="multilabel")
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
    >>> print([round(res, 2) for res in results['roc_auc']])
    [0.83, 0.38, 0.94]
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
class RocAuc(HFMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float")),
                    "references": datasets.Value("int32"),
                }
                if self.config_name == "multiclass"
                else {
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
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        sample_weight: Optional[List[float]] = None,
        average: str = "macro",
        max_fpr: Optional[float] = None,
        multi_class: str = "raise",
        labels: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        score = roc_auc_score(
            y_true=references,
            y_score=predictions,
            average=average,
            sample_weight=sample_weight,
            max_fpr=max_fpr,
            multi_class=multi_class,
            labels=labels,
        )
        return {self.tag: score}
