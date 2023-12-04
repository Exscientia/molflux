from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, Type, Union

import numpy as np
from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.models.pyod import (
    PyODClassificationMixin,
    PyODModelBase,
    PyODModelConfig,
)

try:
    from pyod.models.pca import PCA
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("pyod", e) from None


_DESCRIPTION = """
Principal component analysis (PCA) can be used in detecting outliers.
PCA is a linear dimensionality reduction using Singular Value Decomposition
of the data to project it to a lower dimensional space.

In this procedure, covariance matrix of the data can be decomposed to
orthogonal vectors, called eigenvectors, associated with eigenvalues. The
eigenvectors with high eigenvalues capture most of the variance in the
data.

Therefore, a low dimensional hyperplane constructed by k eigenvectors can
capture most of the variance in the data. However, outliers are different
from normal data points, which is more obvious on the hyperplane
constructed by the eigenvectors with small eigenvalues.

Therefore, outlier scores can be obtained as the sum of the projected
distance of a sample on all eigenvectors.
See :cite:`shyu2003novel,aggarwal2015outlier` for details.

Score(X) = Sum of weighted euclidean distance between each sample to the
hyperplane constructed by the selected eigenvectors
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
n_components : int, float, None or string
    Number of components to keep.
    if n_components is not set all components are kept::

        n_components == min(n_samples, n_features)

    if n_components == 'mle' and svd_solver == 'full', Minka\'s MLE is used
    to guess the dimension
    if ``0 < n_components < 1`` and svd_solver == 'full', select the number
    of components such that the amount of variance that needs to be
    explained is greater than the percentage specified by n_components
    n_components cannot be equal to n_features for svd_solver == 'arpack'.

n_selected_components : int, optional (default=None)
    Number of selected principal components
    for calculating the outlier scores. It is not necessarily equal to
    the total number of the principal components. If not set, use
    all principal components.

contamination : float in (0., 0.5), optional (default=0.1)
    The amount of contamination of the data set, i.e.
    the proportion of outliers in the data set. Used when fitting to
    define the threshold on the decision function.

inplace : bool (default False)
    If True, data passed to fit are overwritten and running
    fit(X).transform(X) will not yield the expected results,
    use fit_transform(X) instead.

whiten : bool, optional (default False)
    When True (False by default) the `components_` vectors are multiplied
    by the square root of n_samples and then divided by the singular values
    to ensure uncorrelated outputs with unit component-wise variances.

    Whitening will remove some information from the transformed signal
    (the relative variance scales of the components) but can sometime
    improve the predictive accuracy of the downstream estimators by
    making their data respect some hard-wired assumptions.

svd_solver : string {'auto', 'full', 'arpack', 'randomized'}
    auto :
        the solver is selected by a default policy based on `X.shape` and
        `n_components`: if the input data is larger than 500x500 and the
        number of components to extract is lower than 80% of the smallest
        dimension of the data, then the more efficient 'randomized'
        method is enabled. Otherwise the exact full SVD is computed and
        optionally truncated afterwards.
    full :
        run exact full SVD calling the standard LAPACK solver via
        `scipy.linalg.svd` and select the components by postprocessing
    arpack :
        run SVD truncated to n_components calling ARPACK solver via
        `scipy.sparse.linalg.svds`. It requires strictly
        0 < n_components < X.shape[1]
    randomized :
        run randomized SVD by the method of Halko et al.

tol : float >= 0, optional (default .0)
    Tolerance for singular values computed by svd_solver == 'arpack'.

iterated_power : int >= 0, or 'auto', (default 'auto')
    Number of iterations for the power method computed by
    svd_solver == 'randomized'.

random_state : int, np.random.Generator instance or None, optional (default None)
    If int, random_state is the seed used by the random number generator;
    If Generator instance, random_state is the random number generator;
    If None, the random number generator is the Generator instance used
    by `np.random`. Used when ``svd_solver`` == 'arpack' or 'randomized'.

weighted : bool, optional (default=True)
    If True, the eigenvalues are used in score computation.
    The eigenvectors with small eigenvalues comes with more importance
    in outlier score calculation.

standardisation : bool, optional (default=True)
    If True, perform standardization first to convert
    data to zero mean and unit variance.
    See http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
"""

Auto = Literal["auto"]
SVDSolver = Literal["auto", "full", "arpack", "randomized"]


class Config:
    extra = "forbid"
    arbitrary_types_allowed = True
    smart_union = True


@dataclass(config=Config)
class PCADetectorConfig(PyODModelConfig):
    n_components: Union[int, float, str, None] = None
    n_selected_components: Optional[int] = None
    contamination: float = 0.1
    inplace: bool = False
    whiten: bool = False
    svd_solver: SVDSolver = "auto"
    tol: float = 0
    iterated_power: Union[int, Auto] = "auto"
    random_state: Union[int, None, np.random.Generator] = None
    weighted: bool = True
    standardisation: bool = True


class PCADetector(PyODClassificationMixin, PyODModelBase[PCADetectorConfig]):
    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[PCADetectorConfig]:
        return PCADetectorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> PCA:
        config = self.model_config
        return PCA(
            n_components=config.n_components,
            n_selected_components=config.n_selected_components,
            contamination=config.contamination,
            copy=not config.inplace,
            whiten=config.whiten,
            svd_solver=config.svd_solver,
            tol=config.tol,
            iterated_power=config.iterated_power,
            random_state=config.random_state,
            weighted=config.weighted,
            standardization=config.standardisation,
        )
