from typing import Any, Dict, Protocol, Tuple, Union, runtime_checkable

from molflux.modelzoo.typing import Classes, DataFrameLike, Features, PredictionResult


@runtime_checkable
class Estimator(Protocol):
    """The public protocol for exs-modelzoo Models."""

    def __init__(self, **kwargs: Any):
        """Initialises the model."""

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model metadata."""

    @property
    def name(self) -> str:
        """The canonical name of the model."""

    @property
    def tag(self) -> str:
        """The arbitrary tag name of the model."""

    @property
    def config(self) -> Dict[str, Any]:
        """The essential config that fully defines the model."""

    @property
    def x_features(self) -> Features:
        """The features that the model has been trained on."""

    @property
    def y_features(self) -> Features:
        """The features that the model predicts."""

    def train(
        self,
        train_data: Union[DataFrameLike, Dict[str, DataFrameLike]],
        **kwargs: Any,
    ) -> Any:
        """Trains the model."""

    def predict(self, data: DataFrameLike, **kwargs: Any) -> PredictionResult:
        """Performs a model prediction."""

    def as_dir(self, directory: str) -> None:
        """Serialises the backend model objects in a directory."""

    def from_dir(self, directory: str) -> None:
        """Deserialises and sets the backend model objects serialised by `as_dir`."""


@runtime_checkable
class SupportsClassification(Protocol):
    @property
    def classes(self) -> Classes:
        """Returns the model's unique class labels"""

    def predict_proba(self, data: DataFrameLike, **kwargs: Any) -> PredictionResult:
        """Returns probability estimates for the data."""


@runtime_checkable
class SupportsUncertaintyCalibration(Protocol):
    """The public protocol for Models supporting uncertainty calibrations."""

    def calibrate_uncertainty(self, data: DataFrameLike, **kwargs: Any) -> Any:
        """Calibrates the uncertainty of predictions on a validation dataset"""


@runtime_checkable
class SupportsPredictionInterval(Protocol):
    """The public protocol for Models supporting prediction interval."""

    def predict_with_prediction_interval(
        self,
        data: DataFrameLike,
        confidence: float,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """Performs a model prediction with associated prediction interval."""


@runtime_checkable
class SupportsStandardDeviation(Protocol):
    """The public protocol for Models supporting standard deviation."""

    def predict_with_std(
        self,
        data: DataFrameLike,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """Performs a model prediction with associated standard deviation."""


@runtime_checkable
class SupportsSampling(Protocol):
    """The public protocol for Models supporting sampling."""

    def sample(
        self,
        data: DataFrameLike,
        n_samples: int,
        **kwargs: Any,
    ) -> PredictionResult:
        """Performs a model prediction with associated samples"""


def supports_classification(model: Any) -> bool:
    """Returns True if the given model is a classifier."""
    return isinstance(model, SupportsClassification)


def supports_uncertainty_calibration(model: Any) -> bool:
    """Return True if the given model supports uncertainty calibration."""
    return isinstance(model, SupportsUncertaintyCalibration)


def supports_prediction_interval(model: Any) -> bool:
    """Return True if the given model supports prediction interval"""
    return isinstance(model, SupportsPredictionInterval)


def supports_std(model: Any) -> bool:
    """Return True if the given model supports standard deviation."""
    return isinstance(model, SupportsStandardDeviation)


def supports_sampling(model: Any) -> bool:
    """Return True if the given model supports sampling."""
    return isinstance(model, SupportsSampling)


@runtime_checkable
class Classifier(Estimator, SupportsClassification, Protocol):
    ...


@runtime_checkable
class UncertaintyEstimator(
    Estimator,
    SupportsUncertaintyCalibration,
    SupportsPredictionInterval,
    Protocol,
):
    ...


Model = Union[Estimator, Classifier, UncertaintyEstimator]
Models = Dict[str, Model]
