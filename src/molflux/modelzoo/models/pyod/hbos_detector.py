from dataclasses import asdict
from typing import Any, Dict, Literal, Type, Union

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.models.pyod import (
    PyODClassificationMixin,
    PyODModelBase,
    PyODModelConfig,
)

try:
    from pyod.models.hbos import HBOS
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("pyod", e) from None


_DESCRIPTION = """
Histogram- based outlier detection (HBOS) is an efficient unsupervised
method. It assumes the feature independence and calculates the degree
of outlyingness by building histograms. See :cite:`goldstein2012histogram`
for details.

Two versions of HBOS are supported:
- Static number of bins: uses a static number of bins for all features.
- Automatic number of bins: every feature uses a number of bins deemed to
  be optimal according to the Birge-Rozenblac method
  (:cite:`birge2006many`).
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
n_bins : int or string, optional (default=10)
    The number of bins. "auto" uses the birge-rozenblac method for
    automatic selection of the optimal number of bins for each feature.

alpha : float in (0, 1), optional (default=0.1)
    The regularizer for preventing overflow.

tol : float in (0, 1), optional (default=0.5)
    The parameter to decide the flexibility while dealing
    the samples falling outside the bins.

contamination : float in (0., 0.5), optional (default=0.1)
    The amount of contamination of the data set,
    i.e. the proportion of outliers in the data set. Used when fitting to
    define the threshold on the decision function.
"""

BinningMethod = Literal["auto"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class HBOSDetectorConfig(PyODModelConfig):
    n_bins: Union[int, BinningMethod] = 10
    alpha: float = 0.1
    tol: float = 0.5
    contamination: float = 0.1


class HBOSDetector(PyODClassificationMixin, PyODModelBase[HBOSDetectorConfig]):
    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[HBOSDetectorConfig]:
        return HBOSDetectorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> HBOS:
        config = self.model_config
        return HBOS(
            n_bins=config.n_bins,
            alpha=config.alpha,
            tol=config.tol,
            contamination=config.contamination,
        )
