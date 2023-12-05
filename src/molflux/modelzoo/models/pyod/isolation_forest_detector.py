from dataclasses import asdict
from typing import Any, Dict, Literal, Type, Union

import numpy as np
from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.models.pyod import (
    PyODClassificationMixin,
    PyODModelBase,
    PyODModelConfig,
)

try:
    from pyod.models.iforest import IForest
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("pyod", e) from None


_DESCRIPTION = """
Wrapper of scikit-learn Isolation Forest with more functionalities.

The IsolationForest 'isolates' observations by randomly selecting a
feature and then randomly selecting a split value between the maximum and
minimum values of the selected feature.
See :cite:`liu2008isolation,liu2012isolation` for details.

Since recursive partitioning can be represented by a tree structure, the
number of splittings required to isolate a sample is equivalent to the path
length from the root node to the terminating node.

This path length, averaged over a forest of such random trees, is a
measure of normality and our decision function.

Random partitioning produces noticeably shorter paths for anomalies.
Hence, when a forest of random trees collectively produce shorter path
lengths for particular samples, they are highly likely to be anomalies.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
n_estimators : int, optional (default=100)
    The number of base estimators in the ensemble.

max_samples : int or float, optional (default="auto")
    The number of samples to draw from X to train each base estimator.

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.
        - If "auto", then `max_samples=min(256, n_samples)`.

    If max_samples is larger than the number of samples provided,
    all samples will be used for all trees (no sampling).

contamination : float in (0., 0.5), optional (default=0.1)
    The amount of contamination of the data set, i.e. the proportion
    of outliers in the data set. Used when fitting to define the threshold
    on the decision function.

max_features : int or float, optional (default=1.0)
    The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

bootstrap : bool, optional (default=False)
    If True, individual trees are fit on random subsets of the training
    data sampled with replacement. If False, sampling without replacement
    is performed.

n_jobs : integer, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    If -1, then the number of jobs is set to the number of cores.

random_state : int, random.Generator instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If Generator instance, random_state is the random number generator;
    If None, the random number generator is the Generator instance used
    by `np.random`.

verbose : int, optional (default=0)
    Controls the verbosity of the tree building process.
"""

Auto = Literal["auto"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class IsolationForestDetectorConfig(PyODModelConfig):
    n_estimators: int = 100
    max_samples: Union[float, int, Auto] = "auto"
    contamination: float = 0.1
    max_features: float = 1.0
    bootstrap: bool = False
    n_jobs: int = 1
    random_state: Union[int, np.random.Generator, None] = None
    verbose: int = 0


class IsolationForestDetector(
    PyODClassificationMixin,
    PyODModelBase[IsolationForestDetectorConfig],
):
    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[IsolationForestDetectorConfig]:
        return IsolationForestDetectorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> IForest:
        config = self.model_config
        return IForest(
            n_estimators=config.n_estimators,
            max_samples=config.max_samples,
            contamination=config.contamination,
            max_features=config.max_features,
            bootstrap=config.bootstrap,
            n_jobs=config.n_jobs,
            behaviour="new",
            random_state=config.random_state,
            verbose=config.verbose,
        )
