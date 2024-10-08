import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

import evaluate
import numpy as np

import datasets
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)


class HFMetric(evaluate.Metric, ABC):
    """A base class for metrics based on the evaluate.Metric API."""

    def __init__(
        self,
        tag: str | None = None,
        config_name: str | None = None,
        keep_in_memory: bool = False,
        seed: int | None = None,
        experiment_id: str | None = None,
        download_config: datasets.DownloadConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialise the evaluate.Metric
        super().__init__(
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            cache_dir=None,
            num_process=1,
            process_id=0,
            seed=seed,
            experiment_id=experiment_id,
            max_concurrent_cache_files=10000,
            timeout=100,
            **kwargs,
        )

        # Download and prepare resources for the metric
        self.download_and_prepare(download_config=download_config)

        self._tag = tag or self.name

        # Initialise a null state
        self._state: dict[str, Any] = {}

        # The metric-specific compute signature
        self._signature = inspect.signature(self._score)

    def __bool__(self) -> bool:
        return True

    def __len__(self) -> int:
        length: int = super().__len__()
        return length

    def __str__(self) -> str:
        return (
            f"Metric(\n"
            f'\tname: "{self.name}",\n'
            f'\ttag: "{self.tag}",\n'
            f"\tfeatures: {self.features},\n"
            f"\tsignature: self.compute{self._signature},\n"
            f'\tdescription: """{self.description}""",\n'
            f'\tusage: """{self.inputs_description}"""\n'
            f"\tstate: {self.state!r}\n"
            ")"
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return asdict(super().info)

    @abstractmethod
    def _info(self) -> evaluate.MetricInfo:
        """Initialises the MetricInfo metadata object for the given metric."""

    @property
    def name(self) -> str:
        metric_name: str = super().name
        return metric_name

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def add_batch(
        self,
        *,
        predictions: ArrayLike | None = None,
        references: ArrayLike | None = None,
        **kwargs: Any,
    ) -> None:
        """Add a batch of predictions and references for the metric's stack."""
        predictions, references = self._pre_process_inputs(
            predictions=predictions,
            references=references,
        )
        super().add_batch(predictions=predictions, references=references, **kwargs)

    def compute(
        self,
        *,
        predictions: ArrayLike | None = None,
        references: ArrayLike | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Computes the metric.
        Args:
            predictions:
            references:
            **kwargs: Can include a 'mask_by_references' kwarg that allows for masking
                the inputs by the valid references values

        Returns:

        """

        # Merge explicit keyword arguments with those stored in config
        if self._state is not None:
            kwargs = {**self._state, **kwargs}

        if "mask_by_references" in kwargs:
            mask_by_references = kwargs.pop("mask_by_references")
        else:
            mask_by_references = False

        # Safeguard against invalid kwargs
        if not all(k in self._signature.parameters for k in kwargs):
            unknown_kwargs = [k for k in kwargs if k not in self._signature.parameters]
            raise ValueError(
                f"Unknown compute parameter(s): {unknown_kwargs}\n\n"
                f"Expected signature self.compute{self._signature}",
            )

        if mask_by_references:
            predictions, references = self._mask_by_references(
                predictions=predictions,
                references=references,
            )

        predictions, references = self._pre_process_inputs(
            predictions=predictions,
            references=references,
        )

        # wrap huggingface's implementation
        try:
            # This loads the data from the cache into self.data,
            # and then calls ._compute()
            result: MetricResult | None = super().compute(
                predictions=predictions,
                references=references,
                **kwargs,
            )

            # On distributed systems, huggingface would return None
            if result is None:
                raise RuntimeError

        except Exception as exc:
            raise RuntimeError(f"Error computing metric {self.tag!r}") from exc

        return result

    def _compute(
        self,
        *,
        predictions: ArrayLike | None = None,
        references: ArrayLike | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """The callable invoked by .compute() to compute the metric."""

        # Do some checks common to all metrics
        if predictions is None or references is None:
            raise RuntimeError(
                f"""Could not compute metric {self.tag}: no data found in the stack.""",
            )

        return self._score(
            predictions=predictions,
            references=references,
            **kwargs,
        )

    @abstractmethod
    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        **kwargs: Any,
    ) -> MetricResult:
        """The callable invoked to compute the metric. To be implemented by subclasses."""

    def reset_state(self) -> None:
        """Resets the state."""
        self._state = {}

    def update_state(self, **kwargs: Any) -> None:
        """Pre-configures keyword arguments for compute()."""
        self._state = {**self._state, **kwargs}

    def _mask_by_references(
        self,
        predictions: ArrayLike | None = None,
        references: ArrayLike | None = None,
    ) -> tuple[ArrayLike | None, ...]:
        if (references is not None) and (predictions is not None):
            # mask if None or NaN
            mask = [not (x is None or np.isnan(x)) for x in references]
            references = [
                x for valid, x in zip(mask, references, strict=False) if valid
            ]
            predictions = [
                x for valid, x in zip(mask, predictions, strict=False) if valid
            ]

        return predictions, references

    def _pre_process_inputs(
        self,
        predictions: ArrayLike | None = None,
        references: ArrayLike | None = None,
    ) -> tuple[ArrayLike | None, ...]:
        """An optional callable used to process inputs before validation.

        For most use cases, there is no need to overwrite this function.
        Defaults to pass-through.
        """
        return predictions, references


class UncertaintyMetric(HFMetric):
    """
    A class for uncertainty metrics.
    Same as HFMetric but with a different signature (eg. for .compute()) so can take in prediction intervals
    """

    def __init__(
        self,
        tag: str | None = None,
        config_name: str | None = None,
        keep_in_memory: bool = False,
        seed: int | None = None,
        experiment_id: str | None = None,
        download_config: datasets.DownloadConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Initialise the HFMetric
        super().__init__(
            tag=tag,
            config_name=config_name,
            keep_in_memory=keep_in_memory,
            seed=seed,
            experiment_id=experiment_id,
            download_config=download_config,
            **kwargs,
        )

    def compute(
        self,
        *,
        predictions: ArrayLike | None = None,
        references: ArrayLike | None = None,
        standard_deviations: ArrayLike | None = None,
        prediction_intervals: ArrayLike | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Computes the metric."""

        # Merge explicit keyword arguments with those stored in config
        if self._state is not None:
            kwargs = {**self._state, **kwargs}

        # Safeguard against invalid kwargs
        if not all(k in self._signature.parameters for k in kwargs):
            unknown_kwargs = [k for k in kwargs if k not in self._signature.parameters]
            raise ValueError(
                f"Unknown compute parameter(s): {unknown_kwargs}\n\n"
                f"Expected signature self.compute{self._signature}",
            )

        predictions, references = self._pre_process_inputs(
            predictions=predictions,
            references=references,
        )

        # wrap huggingface's implementation
        try:
            # This loads the data from the cache into self.data,
            # and then calls ._compute()
            result: MetricResult | None = super().compute(
                predictions=predictions,
                references=references,
                standard_deviations=standard_deviations,
                prediction_intervals=prediction_intervals,
                **kwargs,
            )

            # On distributed systems, huggingface would return None
            if result is None:
                raise RuntimeError

        except Exception as exc:
            raise RuntimeError(f"Error computing metric {self.tag!r}") from exc

        return result

    def _compute(
        self,
        *,
        predictions: ArrayLike | None = None,
        references: ArrayLike | None = None,
        standard_deviations: ArrayLike | None = None,
        prediction_intervals: ArrayLike | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """The callable invoked by .compute() to compute the metric."""

        # Do some checks common to all metrics
        if predictions is None or references is None:
            raise RuntimeError(
                f"""Could not compute metric {self.tag}: no data found in the stack.""",
            )

        return self._score(
            predictions=predictions,
            references=references,
            standard_deviations=standard_deviations,
            prediction_intervals=prediction_intervals,
            **kwargs,
        )

    @abstractmethod
    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        standard_deviations: ArrayLike | None = None,
        prediction_intervals: ArrayLike | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """The callable invoked to compute the metric. To be implemented by subclasses."""
