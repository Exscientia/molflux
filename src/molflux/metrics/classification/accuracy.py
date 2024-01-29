# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Accuracy metric."""

import logging
from typing import Any, List, Optional

import evaluate
from sklearn.metrics import accuracy_score

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Accuracy is the proportion of correct predictions among the total number of cases processed. It can be computed with:
Accuracy = (TP + TN) / (TP + TN + FP + FN)
TP: True positive
TN: True negative
FP: False positive
FN: False negative
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated targets as returned by a classifier.
    references: Ground truth (correct) target values.
    normalize: If False, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
    sample_weight: Sample weights.
Returns:
    accuracy: Accuracy score.
Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("accuracy")
    >>> predictions = [0, 1]
    >>> references = [0, 1]
    >>> metric.compute(predictions=predictions, references=references)
    {'accuracy': 1.0}
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
class Accuracy(HFMetric):
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
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        sample_weight: Optional[List[float]] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        score = accuracy_score(
            y_true=references,
            y_pred=predictions,
            normalize=normalize,
            sample_weight=sample_weight,
        )
        return {self.tag: float(score)}
