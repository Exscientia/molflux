import logging
from dataclasses import asdict, field
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union

from pydantic import validator
from pydantic.dataclasses import dataclass

from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.lightning.trainer.callbacks.stock_callbacks import (
    AVAILABLE_CALLBACKS,
)
from molflux.modelzoo.models.lightning.trainer.schedulers.stock_schedulers import (
    AVAILABLE_SCHEDULERS,
)

try:
    from lightning.pytorch import Trainer
    from lightning.pytorch import callbacks as pl_callbacks
    from lightning.pytorch import loggers as pl_loggers
    from lightning.pytorch import profilers as pl_profilers
    from lightning.pytorch import strategies as pl_strategies
    from torch.optim import Optimizer
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning", e) from e


def make_dvc_logger(**kwargs: Any) -> Any:
    try:
        from dvclive.lightning import DVCLiveLogger  # pyright: ignore

        logging.getLogger("dvc_studio_client.post_live_metrics").setLevel(
            logging.WARNING,
        )

    except ImportError as err:
        from exs.modelzoo.errors import OptionalDependencyImportError

        raise OptionalDependencyImportError("DVCLive", "dvclive") from err

    return DVCLiveLogger(**kwargs)


AVAILABLE_LOGGERS: Dict[str, Callable] = {
    "CometLogger": pl_loggers.CometLogger,
    "CSVLogger": pl_loggers.CSVLogger,
    "DVCLiveLogger": make_dvc_logger,
    "MLFlowLogger": pl_loggers.MLFlowLogger,
    "NeptuneLogger": pl_loggers.NeptuneLogger,
    "TensorBoardLogger": pl_loggers.TensorBoardLogger,
    "WandbLogger": pl_loggers.WandbLogger,
}

AVAILABLE_PROFILERS = {
    "SimpleProfiler": pl_profilers.SimpleProfiler,
    "AdvancedProfiler": pl_profilers.AdvancedProfiler,
    "PyTorchProfiler": pl_profilers.PyTorchProfiler,
}


AVAILABLE_STRATEGIES = {
    "FSDPStrategy": pl_strategies.FSDPStrategy,
    "DDPStrategy": pl_strategies.DDPStrategy,
    "DeepSpeedStrategy": pl_strategies.DeepSpeedStrategy,
}


class ConfigDict:
    extra = "forbid"
    arbitrary_types_allowed = True
    smart_union = True


def _dict_is_single_logger_config(logger_dict: Dict[str, Any]) -> bool:
    return (
        ("name" in logger_dict)
        and ("config" in logger_dict)
        and (isinstance(logger_dict["name"], str))
    )


@dataclass(config=ConfigDict)
class TrainerConfig:
    """
    Lightning Trainer config.
    See https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api for more info.
    """

    accelerator: str = "auto"
    accumulate_grad_batches: int = 1
    barebones: bool = False
    benchmark: Optional[bool] = None
    callbacks: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]], None] = None
    check_val_every_n_epoch: Optional[int] = 1
    default_root_dir: Optional[str] = "training"
    detect_anomaly: bool = False
    deterministic: Union[bool, Literal["warn"], None] = None
    devices: Union[List[int], str, int] = "auto"
    enable_checkpointing: bool = False  # NOTE vanilla PL default is True
    enable_model_summary: bool = True
    enable_progress_bar: bool = True
    fast_dev_run: Union[int, bool] = False
    gradient_clip_algorithm: Optional[str] = None
    gradient_clip_val: Optional[Union[int, float]] = None
    inference_mode: bool = True
    limit_predict_batches: Union[int, float, None] = None
    limit_test_batches: Union[int, float, None] = None
    limit_train_batches: Union[int, float, None] = None
    limit_val_batches: Union[int, float, None] = None
    log_every_n_steps: int = 50
    logger: Union[
        Dict[str, Any],
        List[Dict[str, Any]],
        Dict[str, Dict[str, Any]],
        bool,
        None,
    ] = True
    max_epochs: Optional[int] = 1
    max_steps: int = -1
    max_time: Union[str, Dict[str, int], None] = None
    min_epochs: Optional[int] = None
    min_steps: Optional[int] = None
    num_nodes: int = 1
    num_sanity_val_steps: Optional[int] = None
    overfit_batches: Union[int, float] = 0.0
    precision: Union[int, str] = 32
    profiler: Optional[Dict[str, Any]] = None
    reload_dataloaders_every_n_epochs: int = 0
    strategy: Union[str, Dict[str, Any]] = "auto"
    sync_batchnorm: bool = False
    use_distributed_sampler: bool = True
    val_check_interval: Union[int, float, None] = None

    def pass_to_trainer(self) -> Dict[str, Any]:
        out = asdict(self)
        out["callbacks"] = self.convert_callbacks(out["callbacks"])
        out["logger"] = self.convert_logger(out["logger"])
        out["profiler"] = self.convert_profiler(out["profiler"])
        out["strategy"] = self.convert_strategy(out["strategy"])

        return out

    def convert_callbacks(
        self,
        callbacks: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]], None],
    ) -> Optional[List[pl_callbacks.Callback]]:
        if callbacks is None:
            return callbacks

        if isinstance(callbacks, dict):
            callbacks = list(callbacks.values())

        return [
            AVAILABLE_CALLBACKS[callback["name"]](**callback.get("config", {}))
            for callback in callbacks
        ]

    def convert_logger(
        self,
        logger: Union[
            Dict[str, Any],
            List[Dict[str, Any]],
            Dict[str, Dict[str, Any]],
            bool,
            None,
        ],
    ) -> Union[List[pl_loggers.Logger], bool, None]:
        if logger is None or isinstance(logger, bool):
            return logger

        if isinstance(logger, dict):
            if _dict_is_single_logger_config(logger):
                logger = [logger]
            else:
                logger = list(logger.values())

        return [
            AVAILABLE_LOGGERS[logger_["name"]](**logger_.get("config", {}))
            for logger_ in logger
        ]

    def convert_profiler(
        self,
        profiler: Optional[Dict[str, Any]],
    ) -> Optional[pl_profilers.Profiler]:
        if profiler is None:
            return profiler

        out: pl_profilers.Profiler = AVAILABLE_PROFILERS[profiler["name"]](
            **profiler.get("config", {}),
        )
        return out

    def convert_strategy(
        self,
        strategy: Union[Dict[str, Any], str],
    ) -> Union[pl_strategies.Strategy, str]:
        if isinstance(strategy, str):
            return strategy

        out: pl_strategies.Strategy = AVAILABLE_STRATEGIES[strategy["name"]](
            **strategy.get("config", {}),
        )
        return out


@dataclass(config=ConfigDict)
class OptimizerConfig:
    name: str = "Adam"
    config: Dict[str, Any] = field(default_factory=lambda: {"lr": 1e-4})


@dataclass(config=ConfigDict)
class SchedulerConfig:
    name: str = "CosineAnnealingLR"
    config: Dict[str, Any] = field(default_factory=lambda: {"T_max": "num_steps"})
    interval: Literal["epoch", "step"] = "step"
    frequency: int = 1
    monitor: str = "val_loss"
    strict: bool = True

    def prepare_scheduler(
        self,
        optimizer: Optimizer,
        trainer: Trainer,
    ) -> Dict[str, Any]:
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
            "scheduler": AVAILABLE_SCHEDULERS[self.name](optimizer, **self.config),
            "interval": self.interval,
            "frequency": self.frequency,
            "monitor": self.monitor,
            "strict": self.strict,
        }


@dataclass(config=ConfigDict)
class TransferLearningStage:
    freeze_modules: Optional[List[str]] = None
    trainer: Optional[Dict[str, Any]] = None
    datamodule: Optional[Dict[str, Any]] = None
    optimizer: Optional[Dict[str, Any]] = None
    scheduler: Optional[Dict[str, Any]] = None

    @validator("trainer")
    def must_fit_into_trainer(
        cls,
        trainer: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if trainer is not None:
            TrainerConfig(**trainer)
        return trainer

    @validator("datamodule")
    def must_fit_into_datamodule(
        cls,
        datamodule: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if datamodule is not None:
            DataModuleConfig(**datamodule)
        return datamodule

    @validator("optimizer")
    def must_fit_into_optimizer(
        cls,
        optimizer: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if optimizer is not None:
            OptimizerConfig(**optimizer)
        return optimizer

    @validator("scheduler")
    def must_fit_into_scheduler(
        cls,
        scheduler: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if scheduler is not None:
            SchedulerConfig(**scheduler)
        return scheduler


@dataclass(config=ConfigDict)
class TransferLearningConfigBase:
    stages: List[TransferLearningStage]
    pre_trained_model_path: Optional[str] = None
    repo_url: Optional[str] = None
    rev: Optional[str] = None
    model_path_in_repo: str = "model"
    modules_to_match: Optional[Dict[str, str]] = None

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


@dataclass(config=ConfigDict)
class SplitConfig:
    """batch_size: The batch size to use for this data split, optionally as a
    dictionary assigning different batch sizes to different datasets."""

    batch_size: Union[int, Dict[str, int]] = 1


@dataclass(config=ConfigDict)
class TrainSplitConfig(SplitConfig):
    drop_last: bool = True
    mode: Literal[
        "max_size_cycle",
        "max_size",
        "min_size",
    ] = "max_size_cycle"


@dataclass(config=ConfigDict)
class EvalSplitConfig(SplitConfig):
    mode: Literal[
        "max_size_cycle",
        "max_size",
        "min_size",
        "sequential",
    ] = "max_size"


@dataclass(config=ConfigDict)
class DataModuleConfig:
    train: TrainSplitConfig = field(default_factory=TrainSplitConfig)
    validation: EvalSplitConfig = field(default_factory=EvalSplitConfig)
    test: EvalSplitConfig = field(default_factory=EvalSplitConfig)
    predict: SplitConfig = field(default_factory=SplitConfig)
    num_workers: Union[int, Literal["all"]] = 0


@dataclass(config=ConfigDict)
class CompileConfig:
    mode: Literal["default", "reduce-overhead", "max-autotune"]
    dynamic: bool = False
    fullgraph: bool = False
    backend: str = "inductor"
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(config=ConfigDict)
class LightningConfig(ModelConfig):
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = None
    transfer_learning: Optional[TransferLearningConfigBase] = None
    compile: Union[CompileConfig, bool] = False
    # NOTE This setting affects a global variable!
    float32_matmul_precision: Optional[Literal["highest", "high", "medium"]] = None

    @validator("compile", pre=True)
    def compile_true_to_compile_config_with_default_mode(
        cls,
        compile: Union[Dict[str, Any], bool],
    ) -> Union[Dict[str, Any], bool]:
        if compile is True:
            return {"mode": "default"}
        return compile


LightningConfigT = TypeVar("LightningConfigT", bound=LightningConfig)
