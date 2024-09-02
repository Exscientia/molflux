import logging
from dataclasses import asdict, field
from typing import Any, Literal, TypeVar

from pydantic.v1 import dataclasses, validator

from molflux.modelzoo.model import ModelConfig

try:
    from class_resolver import ClassResolver
    from class_resolver.contrib.torch import lr_scheduler_resolver
    from lightning.pytorch import Trainer
    from lightning.pytorch import callbacks as pl_callbacks
    from lightning.pytorch import loggers as pl_loggers
    from lightning.pytorch import profilers as pl_profilers
    from lightning.pytorch import strategies as pl_strategies
    from torch.optim import Optimizer

    from molflux.modelzoo.models.lightning.trainer.callbacks import callback_resolver
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning", e) from e


logger_resolver = ClassResolver.from_subclasses(
    pl_loggers.Logger,
)

try:
    from dvclive.lightning import DVCLiveLogger  # pyright: ignore

    logger_resolver.register(DVCLiveLogger)

    logging.getLogger("dvc_studio_client.post_live_metrics").setLevel(
        logging.WARNING,
    )

except ImportError:
    pass


profiler_resolver = ClassResolver.from_subclasses(
    pl_profilers.Profiler,
)

strategy_resolver = ClassResolver.from_subclasses(
    pl_strategies.Strategy,
)


class ConfigDict:
    extra = "forbid"
    arbitrary_types_allowed = True
    smart_union = True


def _dict_is_single_logger_config(logger_dict: dict[str, Any]) -> bool:
    return ("name" in logger_dict) and (isinstance(logger_dict["name"], str))


@dataclasses.dataclass(config=ConfigDict)
class TrainerConfig:
    """
    Lightning Trainer config.
    See https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api for more info.
    """

    accelerator: str = "auto"
    accumulate_grad_batches: int = 1
    barebones: bool = False
    benchmark: bool | None = None
    callbacks: list[dict[str, Any]] | dict[str, dict[str, Any]] | None = None
    check_val_every_n_epoch: int | None = 1
    default_root_dir: str | None = "training"
    detect_anomaly: bool = False
    deterministic: bool | Literal["warn"] | None = None
    devices: list[int] | str | int = "auto"
    enable_checkpointing: bool = False  # NOTE vanilla PL default is True
    enable_model_summary: bool = True
    enable_progress_bar: bool = True
    fast_dev_run: int | bool = False
    gradient_clip_algorithm: str | None = None
    gradient_clip_val: int | float | None = None
    inference_mode: bool = True
    limit_predict_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    log_every_n_steps: int = 50
    logger: (
        dict[str, Any] | list[dict[str, Any]] | dict[str, dict[str, Any]] | bool | None
    ) = True
    max_epochs: int | None = 1
    max_steps: int = -1
    max_time: str | dict[str, int] | None = None
    min_epochs: int | None = None
    min_steps: int | None = None
    num_nodes: int = 1
    num_sanity_val_steps: int | None = None
    overfit_batches: int | float = 0.0
    precision: int | str = 32
    profiler: dict[str, Any] | None = None
    reload_dataloaders_every_n_epochs: int = 0
    strategy: str | dict[str, Any] = "auto"
    sync_batchnorm: bool = False
    use_distributed_sampler: bool = True
    val_check_interval: int | float | None = None

    def pass_to_trainer(self) -> dict[str, Any]:
        out = asdict(self)
        out["callbacks"] = self.convert_callbacks(out["callbacks"])
        out["logger"] = self.convert_logger(out["logger"])
        out["profiler"] = self.convert_profiler(out["profiler"])
        out["strategy"] = self.convert_strategy(out["strategy"])

        return out

    def convert_callbacks(
        self,
        callbacks: list[dict[str, Any]] | dict[str, dict[str, Any]] | None,
    ) -> list[pl_callbacks.Callback] | None:
        if callbacks is None:
            return callbacks

        if isinstance(callbacks, dict):
            callbacks = list(callbacks.values())

        return [
            callback_resolver.make(callback["name"], callback.get("config", {}))
            for callback in callbacks
        ]

    def convert_logger(
        self,
        logger: dict[str, Any]
        | list[dict[str, Any]]
        | dict[str, dict[str, Any]]
        | bool
        | None,
    ) -> list[pl_loggers.Logger] | bool | None:
        if logger is None or isinstance(logger, bool):
            return logger

        if isinstance(logger, dict):
            if _dict_is_single_logger_config(logger):
                logger = [logger]
            else:
                logger = list(logger.values())

        return [
            logger_resolver.make(logger_["name"], logger_.get("config", {}))
            for logger_ in logger
        ]

    def convert_profiler(
        self,
        profiler: dict[str, Any] | None,
    ) -> pl_profilers.Profiler | None:
        if profiler is None:
            return profiler

        return profiler_resolver.make(profiler["name"], profiler.get("config", {}))

    def convert_strategy(
        self,
        strategy: dict[str, Any] | str,
    ) -> pl_strategies.Strategy | str:
        if isinstance(strategy, str):
            return strategy

        return strategy_resolver.make(strategy["name"], strategy.get("config", {}))


@dataclasses.dataclass(config=ConfigDict)
class OptimizerConfig:
    name: str = "Adam"
    config: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-4})


@dataclasses.dataclass(config=ConfigDict)
class SchedulerConfig:
    name: str = "CosineAnnealingLR"
    config: dict[str, Any] = field(default_factory=lambda: {"T_max": "num_steps"})
    interval: Literal["epoch", "step"] = "step"
    frequency: int = 1
    monitor: str = "val_loss"
    strict: bool = True

    def prepare_scheduler(
        self,
        optimizer: Optimizer,
        trainer: Trainer,
    ) -> dict[str, Any]:
        # NOTE Enables config-driven decay of learning rate to final step
        if self.config.get("T_max", None) == "num_steps":
            self.config["T_max"] = trainer.estimated_stepping_batches
            self.interval = "step"
        elif self.config.get("T_max", None) == "num_epochs":
            self.config["T_max"] = trainer.max_epochs
            self.interval = "epoch"

        if self.config.get("total_steps", None) == "num_steps":
            self.config["total_steps"] = trainer.estimated_stepping_batches
            self.interval = "step"
        elif self.config.get("total_steps", None) == "num_epochs":
            self.config["total_steps"] = trainer.max_epochs
            self.interval = "epoch"

        return {
            "scheduler": lr_scheduler_resolver.make(
                self.name,
                self.config,
                optimizer=optimizer,
            ),
            "interval": self.interval,
            "frequency": self.frequency,
            "monitor": self.monitor,
            "strict": self.strict,
        }


@dataclasses.dataclass(config=ConfigDict)
class TransferLearningStage:
    freeze_modules: list[str] | None = None
    trainer: dict[str, Any] | None = None
    datamodule: dict[str, Any] | None = None
    optimizer: dict[str, Any] | None = None
    scheduler: dict[str, Any] | None = None

    @validator("trainer")
    def must_fit_into_trainer(
        cls,
        trainer: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if trainer is not None:
            TrainerConfig(**trainer)
        return trainer

    @validator("datamodule")
    def must_fit_into_datamodule(
        cls,
        datamodule: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if datamodule is not None:
            DataModuleConfig(**datamodule)
        return datamodule

    @validator("optimizer")
    def must_fit_into_optimizer(
        cls,
        optimizer: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if optimizer is not None:
            OptimizerConfig(**optimizer)
        return optimizer

    @validator("scheduler")
    def must_fit_into_scheduler(
        cls,
        scheduler: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if scheduler is not None:
            SchedulerConfig(**scheduler)
        return scheduler


@dataclasses.dataclass(config=ConfigDict)
class TransferLearningConfigBase:
    stages: list[TransferLearningStage]
    pre_trained_model_path: str | None = None
    repo_url: str | None = None
    rev: str | None = None
    model_path_in_repo: str = "model"
    modules_to_match: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if (self.pre_trained_model_path is not None) ^ (
            (self.repo_url is not None)
            and (self.rev is not None)
            and (self.model_path_in_repo is not None)
        ):
            pass
        else:
            raise ValueError(
                "Must specify either 'pre_trained_model_path' or all of 'repo_url', 'rev', 'model_path_in_repo' (but not both).",
            )

        for i in range(len(self.stages)):
            if isinstance(self.stages[i], dict):
                self.stages[i] = TransferLearningStage(**self.stages[i])  # type: ignore


@dataclasses.dataclass(config=ConfigDict)
class SplitConfig:
    """batch_size: The batch size to use for this data split, optionally as a
    dictionary assigning different batch sizes to different datasets."""

    batch_size: int | dict[str, int] = 1


@dataclasses.dataclass(config=ConfigDict)
class TrainSplitConfig(SplitConfig):
    drop_last: bool = True
    mode: Literal[
        "max_size_cycle",
        "max_size",
        "min_size",
    ] = "max_size_cycle"


@dataclasses.dataclass(config=ConfigDict)
class EvalSplitConfig(SplitConfig):
    mode: Literal[
        "max_size_cycle",
        "max_size",
        "min_size",
        "sequential",
    ] = "max_size"


@dataclasses.dataclass(config=ConfigDict)
class DataModuleConfig:
    train: TrainSplitConfig = field(default_factory=TrainSplitConfig)
    validation: EvalSplitConfig = field(default_factory=EvalSplitConfig)
    test: EvalSplitConfig = field(default_factory=EvalSplitConfig)
    predict: SplitConfig = field(default_factory=SplitConfig)
    num_workers: int | Literal["all"] = 0


@dataclasses.dataclass(config=ConfigDict)
class CompileConfig:
    mode: Literal["default", "reduce-overhead", "max-autotune"]
    dynamic: bool = False
    fullgraph: bool = False
    backend: str = "inductor"
    backend_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclasses.dataclass(config=ConfigDict)
class LightningConfig(ModelConfig):
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig | None = None
    transfer_learning: TransferLearningConfigBase | None = None
    compile: CompileConfig | bool = False
    # NOTE This setting affects a global variable!
    float32_matmul_precision: Literal["highest", "high", "medium"] | None = None

    @validator("compile", pre=True)
    def compile_true_to_compile_config_with_default_mode(
        cls,
        compile: dict[str, Any] | bool,
    ) -> dict[str, Any] | bool:
        if compile is True:
            return {"mode": "default"}
        return compile


LightningConfigT = TypeVar("LightningConfigT", bound=LightningConfig)
