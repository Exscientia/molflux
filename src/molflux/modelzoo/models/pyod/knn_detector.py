from dataclasses import asdict
from typing import Any, Callable, Dict, Literal, Optional, Type, Union

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.models.pyod import (
    PyODClassificationMixin,
    PyODModelBase,
    PyODModelConfig,
)

try:
    from pyod.models.knn import KNN
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("pyod", e) from None


_DESCRIPTION = """
kNN class for outlier detection.
For an observation, its distance to its kth nearest neighbour could be
viewed as the outlying score. It could be viewed as a way to measure
the density. See :cite:`ramaswamy2000efficient,angiulli2002fast` for
details.

Three kNN detectors are supported:
largest: use the distance to the kth neighbour as the outlier score
mean: use the average of all k neighbours as the outlier score
median: use the median of the distance to k neighbours as the outlier score
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
contamination : float in (0., 0.5), optional (default=0.1)
    The amount of contamination of the data set,
    i.e. the proportion of outliers in the data set. Used when fitting to
    define the threshold on the decision function.

n_neighbours : int, optional (default = 5)
    Number of neighbours to use by default for k neighbours queries.

method : str, optional (default='largest')
    {'largest', 'mean', 'median'}

    - 'largest': use the distance to the kth neighbour as the outlier score
    - 'mean': use the average of all k neighbours as the outlier score
    - 'median': use the median of the distance to k neighbours as the
      outlier score

radius : float, optional (default = 1.0)
    Range of parameter space to use by default for `radius_neighbours`
    queries.

leaf_size : int, optional (default = 30)
    Leaf size passed to BallTree. This can affect the
    speed of the construction and query, as well as the memory
    required to store the tree.  The optimal value depends on the
    nature of the problem.

metric : string or callable, default 'minkowski'
    metric to use for distance computation. Any metric from scikit-learn
    or scipy.spatial.distance can be used.

    If metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays as input and return one value indicating the
    distance between them. This works for Scipy's metrics, but is less
    efficient than passing the metric name as a string.

    Distance matrices are not supported.

    Valid values for metric are:

    - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']

    - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
      'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
      'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
      'sqeuclidean', 'yule']

    See the documentation for scipy.spatial.distance for details on these
    metrics.

p : integer, optional (default = 2)
    Parameter for the Minkowski metric from
    sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances

metric_params : dict, optional (default = None)
    Additional keyword arguments for the metric function.

n_jobs : int, optional (default = 1)
    The number of parallel jobs to run for neighbours search.
    If ``-1``, then the number of jobs is set to the number of CPU cores.
    Affects only kneighbours and kneighbours_graph methods.
"""

Method = Literal["largest", "mean", "median"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class KNNDetectorConfig(PyODModelConfig):
    contamination: float = 0.1
    n_neighbours: int = 5
    method: Method = "largest"
    radius: float = 1.0
    leaf_size: int = 30
    metric: Union[str, Callable] = "minkowski"
    p: int = 2
    metric_params: Optional[Dict] = None
    n_jobs: int = 1


class KNNDetector(PyODClassificationMixin, PyODModelBase[KNNDetectorConfig]):
    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[KNNDetectorConfig]:
        return KNNDetectorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> KNN:
        config = self.model_config
        return KNN(
            contamination=config.contamination,
            n_neighbors=config.n_neighbours,
            method=config.method,
            radius=config.radius,
            algorithm="ball_tree",
            leaf_size=config.leaf_size,
            metric=config.metric,
            p=config.p,
            metric_params=config.metric_params,
            n_jobs=config.n_jobs,
        )
