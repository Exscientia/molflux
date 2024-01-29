"""
Abstract Base Classes for classes implementing the Model protocol.
"""
import inspect
import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import field
from functools import cached_property
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

from pydantic.dataclasses import dataclass

from datasets import Dataset, concatenate_datasets
from molflux.modelzoo.errors import NotTrainedError
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.interchange import hf_dataset_from_dataframe
from molflux.modelzoo.naming import camelcase_to_snakecase
from molflux.modelzoo.typing import (
    Classes,
    DataFrameLike,
    Features,
    PredictionResult,
)
from molflux.modelzoo.utils import (
    pick_features,
    raise_on_unrecognised_parameters,
    validate_features,
    validate_prediction_result_num_tasks,
)
from molflux.version import version as __version__

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    x_features: Features = field(default_factory=list)
    y_features: Features = field(default_factory=list)
    train_features: Union[Features, Dict[str, Features], None] = None

    def resolve_train_features(self, name: Optional[str] = None) -> Features:
        """Returns the features expected in training dataset `name`."""
        if self.train_features is None:
            return self.x_features + self.y_features
        elif isinstance(self.train_features, dict):
            if name is None:
                raise ValueError(
                    "This model's features differ by dataset. Specify its name.",
                )

            return self.train_features.get(name, self.x_features + self.y_features)
        else:
            return self.train_features


_ModelConfigT = TypeVar("_ModelConfigT", bound=ModelConfig)


class ModelBase(Generic[_ModelConfigT], ABC):
    _train_multi_data_enabled: bool = False

    def __init__(self, *, tag: Optional[str] = None, **config_kwargs: Any) -> None:
        """Initialises a new model."""

        # build default config
        self.model_config = self._config_builder(**config_kwargs)

        # build default info (with custom values if specified)
        info = self._info()
        info.name = camelcase_to_snakecase(type(self).__name__)
        info.tag = tag or info.name
        info.config = self.config
        info.version = __version__
        self.info = info

        # The backend model object(s)
        self.model: Any = None

    @property
    def config(self) -> Dict[str, Any]:
        return self.model_config.__dict__

    @cached_property
    def config_signature(self) -> inspect.Signature:
        """The model config signature.

        This is defined as a property to work on
        unpickled objects from previous modelzoo versions.
        """
        return inspect.signature(self.model_config.__class__.__init__)

    @cached_property
    def predict_signature(self) -> inspect.Signature:
        """The predict signature.

        This is defined as a property to be able to run predictions
        on unpickled objects from previous modelzoo versions."""
        return inspect.signature(self._predict)

    @cached_property
    def train_signature(self) -> inspect.Signature:
        """The train signature.

        This is defined as a property to be able to train
        unpickled objects from previous modelzoo versions."""
        return inspect.signature(self._train)

    @property
    def x_features(self) -> Features:
        return self.model_config.x_features

    @property
    def y_features(self) -> Features:
        return self.model_config.y_features

    @property
    def train_features(self) -> Union[Features, Dict[str, Features], None]:
        return self.model_config.train_features

    def __str__(self) -> str:
        return (
            f"Model(\n"
            f'\tname: "{self.name}",\n'
            f'\ttag: "{self.tag}",\n'
            f'\tdescription: """{self.info.model_description}""",\n'
            f"\tconfig signature: __init__{self.config_signature},\n"
            f'\tconfig: """{self.info.config_description}""",\n'
            f"\ttrain signature: self.train{self.train_signature},\n"
            f"\tpredict signature: self.predict{self.predict_signature},\n"
            ")"
        )

    @property
    @abstractmethod
    def _config_builder(self) -> Type[_ModelConfigT]:
        """The callable that initialises the model config object.

        To be implemented by subclasses.
        """

    @abstractmethod
    def _info(self) -> ModelInfo:
        """Initialises the ModelInfo object.

        To be implemented by subclasses.
        """

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.info.to_dict()

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def tag(self) -> str:
        return self.info.tag

    def train(
        self,
        train_data: Union[DataFrameLike, Dict[Optional[str], DataFrameLike]],
        **kwargs: Any,
    ) -> Any:
        """Trains the model."""

        if self._train_multi_data_enabled:
            # if passed a single dataset, convert to dict
            if not isinstance(train_data, dict):
                train_data = {None: train_data}

            # make sure data is a datasets.Dataset
            train_datasets = {
                k: hf_dataset_from_dataframe(v) for k, v in train_data.items()
            }

            # make sure dataset has required features
            for name, dataset in train_datasets.items():
                validate_features(
                    dataset,
                    self.model_config.resolve_train_features(name),
                )

            # Safeguard against invalid kwargs
            raise_on_unrecognised_parameters(self._train_multi_data, **kwargs)

            return self._train_multi_data(train_data=train_datasets, **kwargs)

        else:
            if isinstance(train_data, dict):
                train_dataset = concatenate_datasets(
                    [
                        hf_dataset_from_dataframe(dataset)
                        for dataset in train_data.values()
                    ],
                )
            else:
                train_dataset = hf_dataset_from_dataframe(train_data)

            validate_features(train_dataset, self.model_config.resolve_train_features())

            raise_on_unrecognised_parameters(self._train, **kwargs)

            return self._train(train_data=train_dataset, **kwargs)

    @abstractmethod
    def _train(self, train_data: Dataset, **kwargs: Any) -> Any:
        """The training callable.

        To be implemented by subclasses.
        """

    def _train_multi_data(
        self,
        train_data: Dict[Optional[str], Dataset],
        **kwargs: Any,
    ) -> Any:
        """The training callable for multiple datasets.

        To be implemented by subclasses if they support multiple datasets.
        """
        del train_data, kwargs
        raise NotImplementedError(
            "This class does not support training on multiple datasets.",
        )

    def predict(self, data: DataFrameLike, **kwargs: Any) -> PredictionResult:
        """Performs a model prediction."""

        # make sure data is a datasets.Dataset
        data = hf_dataset_from_dataframe(data)

        # make sure dataset has required features
        validate_features(data, self.x_features)

        # pick out relevant features
        data = pick_features(data, self.x_features)

        # Safeguard against invalid kwargs
        raise_on_unrecognised_parameters(self._predict, **kwargs)

        prediction_result = self._predict(data=data, **kwargs)

        validate_prediction_result_num_tasks(prediction_result, self.y_features)

        return prediction_result

    @abstractmethod
    def _predict(self, data: Dataset, **kwargs: Any) -> PredictionResult:
        """The prediction callable.

        To be implemented by subclasses.
        """

    @property
    def _predict_display_names(self) -> List[str]:
        """The display names for prediction outputs.

        These are not part of the API, we just provide this utility function
        to help making display names consistent across all of our models.
        """
        return [f"{self.tag}::{task}" for task in self.y_features]

    def as_dir(self, directory: str) -> None:
        """Serialises a pre-trained model in a directory."""

        if self.model is None:
            raise NotTrainedError

        pickle_fn = os.path.join(directory, "model.pkl")
        with open(pickle_fn, "wb") as f:
            pickle.dump(self.model, f)

    def from_dir(self, directory: str) -> None:
        """Deserialises the backend model object stored in a given directory.

        This should undo the process defined in `self.as_dir()`.
        """

        # The expected model binary
        filename = os.path.join(directory, "model.pkl")
        with open(filename, "rb") as f:
            self.model = pickle.load(f)  # noqa: S301
            return


class ClassificationMixin(ABC):
    """Mixin for all classifiers in exs-modelzoo."""

    tag: str
    x_features: Features
    y_features: Features

    @property
    @abstractmethod
    def classes(self) -> Classes:
        """Dictionary mapping each task name to a list of possible classes"""
        ...

    @cached_property
    def predict_proba_signature(self) -> inspect.Signature:
        """The predict_proba signature."""
        return inspect.signature(self._predict_proba)

    def predict_proba(self, data: DataFrameLike, **kwargs: Any) -> PredictionResult:
        """Returns probability estimates for the data."""

        # make sure data is a datasets.Dataset
        data = hf_dataset_from_dataframe(data)

        # make sure dataset has required features
        validate_features(data, self.x_features)

        # pick out relevant features
        data = pick_features(data, self.x_features)

        # Safeguard against invalid kwargs
        raise_on_unrecognised_parameters(self._predict_proba, **kwargs)

        prediction_result = self._predict_proba(data=data, **kwargs)

        validate_prediction_result_num_tasks(prediction_result, self.y_features)

        return prediction_result

    @abstractmethod
    def _predict_proba(self, data: Dataset, **kwargs: Any) -> PredictionResult:
        """The predict_proba callable.

        To be implemented by subclasses.
        """

    @property
    def _predict_proba_display_names(self) -> List[str]:
        """The display names for prediction outputs.

        These are not part of the API, we just provide this utility function
        to help making display names consistent across all of our models.
        """
        return [f"{self.tag}::{task}::probabilities" for task in self.y_features]

    def __str__(self) -> str:
        return super().__str__() + (
            f"\nClassifier(\n"
            f'\tclasses: "{self.classes}",\n'
            f"\tpredict_proba signature: self.predict_proba{self.predict_proba_signature},\n"
            ")"
        )


class UncertaintyCalibrationMixin(ABC):
    """Mixin for all models supporting uncertainty calibration exs-modelzoo"""

    x_features: Features
    y_features: Features

    @cached_property
    def calibrate_uncertainty_signature(self) -> inspect.Signature:
        """The calibrate_uncertainty signature."""
        return inspect.signature(self._calibrate_uncertainty)

    def calibrate_uncertainty(self, data: DataFrameLike, **kwargs: Any) -> Any:
        """Calibrates the uncertainty of predictions on a validation dataset"""

        # make sure data is a datasets.Dataset
        data = hf_dataset_from_dataframe(data)

        # make sure dataset has required features
        # y features are not validated as not all models might define them
        validate_features(data, self.x_features)

        # Safeguard against invalid kwargs
        raise_on_unrecognised_parameters(self._calibrate_uncertainty, **kwargs)

        # idea: could rearrange here the predictions and names depending on the format
        return self._calibrate_uncertainty(data=data, **kwargs)

    @abstractmethod
    def _calibrate_uncertainty(self, data: Dataset, **kwargs: Any) -> Any:
        """The uncertainty calibration callable.

        To be implemented by subclasses.
        """

    def __str__(self) -> str:
        return super().__str__() + (
            f"\nEstimator(\n"
            f"\tcalibrate_uncertainty signature: self.calibrate_uncertainty_signature{self.calibrate_uncertainty_signature},\n"
            ")"
        )


class PredictionIntervalMixin(ABC):
    """Mixin for all models supporting prediction intervals."""

    tag: str
    x_features: Features
    y_features: Features

    @cached_property
    def predict_with_prediction_interval_signature(self) -> inspect.Signature:
        return inspect.signature(self._predict_with_prediction_interval)

    def predict_with_prediction_interval(
        self,
        data: DataFrameLike,
        confidence: float,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """Return a model prediction with an associated prediction interval"""

        # make sure data is a datasets.Dataset
        data = hf_dataset_from_dataframe(data)

        # make sure dataset has required features
        validate_features(data, self.x_features)

        # pick out relevant features
        data = pick_features(data, self.x_features)

        # Safeguard against invalid kwargs
        raise_on_unrecognised_parameters(
            self._predict_with_prediction_interval,
            **kwargs,
        )

        (
            prediction_result,
            prediction_prediction_interval_result,
        ) = self._predict_with_prediction_interval(
            data,
            confidence=confidence,
            **kwargs,
        )

        validate_prediction_result_num_tasks(prediction_result, self.y_features)
        validate_prediction_result_num_tasks(
            prediction_prediction_interval_result,
            self.y_features,
        )

        return prediction_result, prediction_prediction_interval_result

    @abstractmethod
    def _predict_with_prediction_interval(
        self,
        data: Dataset,
        confidence: float,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """The prediction interval prediction callable for prediction intervals.

        To be implemented by subclasses.
        """

    @property
    def _predict_with_prediction_interval_display_names(
        self,
    ) -> Tuple[List[str], List[str]]:
        """The display names for prediction outputs.

        These are not part of the API, we just provide this utility function
        to help making display names consistent across all of our models.
        """
        return zip(  # type: ignore[return-value]
            *[
                (f"{self.tag}::{task}", f"{self.tag}::{task}::prediction_interval")
                for task in self.y_features
            ],
        )

    def __str__(self) -> str:
        return super().__str__() + (
            f"\nEstimator(\n"
            f"\tpredict_with_prediction_interval signature: self.predict_with_prediction_interval{self.predict_with_prediction_interval},\n"
            ")"
        )


class StandardDeviationMixin(ABC):
    """Mixin for all models supporting predictions with standard deviations"""

    tag: str
    x_features: Features
    y_features: Features

    @cached_property
    def predict_with_std_signature(self) -> inspect.Signature:
        return inspect.signature(self._predict_with_std)

    def predict_with_std(
        self,
        data: DataFrameLike,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """Return a model prediction with an associated standard deviation"""

        # make sure data is a datasets.Dataset
        data = hf_dataset_from_dataframe(data)

        # make sure dataset has required features
        validate_features(data, self.x_features)

        # pick out relevant features
        data = pick_features(data, self.x_features)

        # Safeguard against invalid kwargs
        raise_on_unrecognised_parameters(self._predict_with_std, **kwargs)

        prediction_result, prediction_std_result = self._predict_with_std(
            data,
            **kwargs,
        )

        validate_prediction_result_num_tasks(prediction_result, self.y_features)
        validate_prediction_result_num_tasks(prediction_std_result, self.y_features)

        return prediction_result, prediction_std_result

    @abstractmethod
    def _predict_with_std(
        self,
        data: Dataset,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """The prediction callable for prediction standard deviations.

        To be implemented by subclasses.
        """

    @property
    def _predict_with_std_display_names(self) -> Tuple[List[str], List[str]]:
        """The display names for prediction outputs.

        These are not part of the API, we just provide this utility function
        to help making display names consistent across all of our models.
        """
        return zip(  # type: ignore[return-value]
            *[
                (f"{self.tag}::{task}", f"{self.tag}::{task}::std")
                for task in self.y_features
            ],
        )

    def __str__(self) -> str:
        return super().__str__() + (
            f"\nEstimator(\n"
            f"\tpredict_with_std signature: self.predict_with_std{self.predict_with_std_signature},\n"
            ")"
        )


class CovarianceMixin(ABC):
    """Mixin for all models supporting predictions with a covariance matrix of joint predictions"""

    tag: str
    x_features: Features
    y_features: Features

    @cached_property
    def predict_with_covariance_signature(self) -> inspect.Signature:
        return inspect.signature(self._predict_with_covariance)

    def predict_with_covariance(
        self,
        data: DataFrameLike,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """Return a model prediction with an associated standard deviation"""

        # make sure data is a datasets.Dataset
        data = hf_dataset_from_dataframe(data)

        # make sure dataset has required features
        validate_features(data, self.x_features)

        # pick out relevant features
        data = pick_features(data, self.x_features)

        # Safeguard against invalid kwargs
        raise_on_unrecognised_parameters(self._predict_with_covariance, **kwargs)

        prediction_result, prediction_covariance_result = self._predict_with_covariance(
            data,
            **kwargs,
        )

        validate_prediction_result_num_tasks(prediction_result, self.y_features)
        validate_prediction_result_num_tasks(
            prediction_covariance_result,
            self.y_features,
        )

        return prediction_result, prediction_covariance_result

    @abstractmethod
    def _predict_with_covariance(
        self,
        data: Dataset,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """The prediction callable for prediction covariances.

        To be implemented by subclasses.
        """

    @property
    def _predict_with_covariance_display_names(self) -> Tuple[List[str], List[str]]:
        """The display names for prediction outputs.

        These are not part of the API, we just provide this utility function
        to help making display names consistent across all of our models.
        """
        return zip(  # type: ignore[return-value]
            *[
                (f"{self.tag}::{task}", f"{self.tag}::{task}::covariance")
                for task in self.y_features
            ],
        )

    def __str__(self) -> str:
        return super().__str__() + (
            f"\nEstimator(\n"
            f"\tpredict_with_covariance signature: self.predict_with_covariance{self.predict_with_covariance_signature},\n"
            ")"
        )


class SamplingMixin(ABC):
    """Mixin for all models supporting sampling"""

    tag: str
    x_features: Features
    y_features: Features

    @cached_property
    def sample_signature(self) -> inspect.Signature:
        return inspect.signature(self._sample)

    def sample(
        self,
        data: DataFrameLike,
        n_samples: int,
        **kwargs: Any,
    ) -> PredictionResult:
        """Return n_samples prediction estimates with the model"""

        # make sure data is a datasets.Dataset
        data = hf_dataset_from_dataframe(data)

        # make sure dataset has required features
        validate_features(data, self.x_features)

        # pick out relevant features
        data = pick_features(data, self.x_features)

        # Safeguard against invalid kwargs
        raise_on_unrecognised_parameters(self._sample, n_samples=n_samples, **kwargs)

        sampling_result = self._sample(data, n_samples=n_samples, **kwargs)

        validate_prediction_result_num_tasks(sampling_result, self.y_features)

        return sampling_result

    @abstractmethod
    def _sample(
        self,
        data: Dataset,
        n_samples: int,
        **kwargs: Any,
    ) -> PredictionResult:
        """The samples callable, generating n_samples predictions for the input.

        To be implemented by subclasses.
        """

    @property
    def _sample_display_names(self) -> List[str]:
        """The display names for prediction outputs.

        These are not part of the API, we just provide this utility function
        to help making display names consistent across all of our models.
        """
        return [f"{self.tag}::{task}::samples" for task in self.y_features]

    def __str__(self) -> str:
        return super().__str__() + (
            f"\nEstimator(\n"
            f"\tsample signature: self.sample{self.sample_signature},\n"
            ")"
        )
