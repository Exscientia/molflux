from typing import Any, Literal, Optional, Type

from pydantic.dataclasses import dataclass

import datasets
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import (
    SKLearnClassificationMixin,
    SKLearnModelBase,
)

try:
    from sklearn.neighbors import KNeighborsClassifier
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
This is an sklearn K nearest neighbours classifier.

Neighbors-based classification is a type of instance-based learning or non-generalizing
learning: it does not attempt to construct a general internal model, but simply stores
instances of the training data. Classification is computed from a simple majority vote
of the nearest neighbors of each point: a query point is assigned the data class which
has the most representatives within the nearest neighbors of the point.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
n_neighbors : int, default=5
    Number of neighbors to use by default for :meth:`kneighbors` queries.
weights : {'uniform', 'distance'} or callable, default='uniform'
    Weight function used in prediction.  Possible values:
    - 'uniform' : uniform weights.  All points in each neighborhood
      are weighted equally.
    - 'distance' : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
    - [callable] : a user-defined function which accepts an
      array of distances, and returns an array of the same shape
      containing the weights.
algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
    Algorithm used to compute the nearest neighbors:
    - 'ball_tree' will use :class:`BallTree`
    - 'kd_tree' will use :class:`KDTree`
    - 'brute' will use a brute-force search.
    - 'auto' will attempt to decide the most appropriate algorithm
      based on the values passed to :meth:`fit` method.
    Note: fitting on sparse input will override the setting of
    this parameter, using brute force.
leaf_size : int, default=30
    Leaf size passed to BallTree or KDTree.  This can affect the
    speed of the construction and query, as well as the memory
    required to store the tree.  The optimal value depends on the
    nature of the problem.
p : int, default=2
    Power parameter for the Minkowski metric. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
metric : str or callable, default='minkowski'
    The distance metric to use for the tree.  The default metric is
    minkowski, and with p=2 is equivalent to the standard Euclidean
    metric. For a list of available metrics, see the documentation of
    :class:`~sklearn.metrics.DistanceMetric` and the metrics listed in
    `sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
    "cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.
    If metric is "precomputed", X is assumed to be a distance matrix and
    must be square during fit. X may be a :term:`sparse graph`,
    in which case only "nonzero" molfluxs may be considered neighbors.
metric_params : dict, default=None
    Additional keyword arguments for the metric function.
n_jobs : int, default=None
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors.
    for more details.
    Doesn't affect :meth:`fit` method.
"""
Weights = Literal["uniform", "distance"]
Algorithm = Literal["auto", "ball_tree", "kd_tree", "brute"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class KNNClassifierConfig(ModelConfig):
    n_neighbors: int = 5
    weights: Weights = "distance"
    algorithm: Algorithm = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: str = "minkowski"
    metric_params: Optional[dict] = None
    n_jobs: Optional[int] = None


class KNNClassifier(SKLearnClassificationMixin, SKLearnModelBase[KNNClassifierConfig]):
    @property
    def _config_builder(self) -> Type[KNNClassifierConfig]:
        return KNNClassifierConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> KNeighborsClassifier:
        config = self.model_config
        return KNeighborsClassifier(
            n_neighbors=config.n_neighbors,
            weights=config.weights,
            algorithm=config.algorithm,
            leaf_size=config.leaf_size,
            p=config.p,
            metric=config.metric,
            metric_params=config.metric_params,
            n_jobs=config.n_jobs,
        )

    def _train(
        self,
        train_data: datasets.Dataset,
        **kwargs: Any,
    ) -> Any:
        # perform validation of data
        n_data_points = len(train_data)
        n_neighbors = self.model_config.n_neighbors
        if n_data_points < n_neighbors:
            raise RuntimeError(
                f"The training data has {n_data_points} points, but the n_neighbors "
                f"parameter of the model is {n_neighbors}. Lower the n_neighbors value "
                f"in your model config or use more data!",
            )

        return super()._train(train_data=train_data, **kwargs)
