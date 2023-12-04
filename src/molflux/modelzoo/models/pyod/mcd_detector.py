from dataclasses import asdict
from typing import Any, Dict, Optional, Type, Union

import numpy as np
from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.models.pyod import (
    PyODClassificationMixin,
    PyODModelBase,
    PyODModelConfig,
)

try:
    from pyod.models.mcd import MCD
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("pyod", e) from None


_DESCRIPTION = """
Detecting outliers in a Gaussian distributed dataset using
Minimum Covariance Determinant (MCD): robust estimator of covariance.

The Minimum Covariance Determinant covariance estimator is to be applied
on Gaussian-distributed data, but could still be relevant on data
drawn from a unimodal, symmetric distribution. It is not meant to be used
with multi-modal data (the algorithm used to fit a MinCovDet object is
likely to fail in such a case).
One should consider projection pursuit methods to deal with multi-modal
datasets.

First fit a minimum covariance determinant model and then compute the
Mahalanobis distance as the outlier degree of the data

See :cite:`rousseeuw1999fast,hardin2004outlier` for details.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
contamination : float in (0., 0.5), optional (default=0.1)
    The amount of contamination of the data set,
    i.e. the proportion of outliers in the data set. Used when fitting to
    define the threshold on the decision function.

store_precision : bool
    Specify if the estimated precision is stored.

assume_centered : bool
    If True, the support of the robust location and the covariance
    estimates is computed, and a covariance estimate is recomputed from
    it, without centering the data.
    Useful to work with data whose mean is significantly equal to
    zero but is not exactly zero.
    If False, the robust location and covariance are directly computed
    with the FastMCD algorithm without additional treatment.

support_fraction : float, 0 < support_fraction < 1
    The proportion of points to be included in the support of the raw
    MCD estimate. Default is None, which implies that the minimum
    value of support_fraction will be used within the algorithm:
    [n_sample + n_features + 1] / 2

random_state : int, np.random.Generator instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If Generator instance, random_state is the random number generator;
    If None, the random number generator is the Generator instance used
    by `np.random`.
"""


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class MCDDetectorConfig(PyODModelConfig):
    contamination: float = 0.1
    store_precision: bool = True
    assume_centered: bool = False
    support_fraction: Optional[float] = None
    random_state: Union[int, np.random.Generator, None] = None


class MCDDetector(PyODClassificationMixin, PyODModelBase[MCDDetectorConfig]):
    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[MCDDetectorConfig]:
        return MCDDetectorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> MCD:
        config = self.model_config
        return MCD(
            contamination=config.contamination,
            store_precision=config.store_precision,
            assume_centered=config.assume_centered,
            support_fraction=config.support_fraction,
            random_state=config.random_state,
        )
