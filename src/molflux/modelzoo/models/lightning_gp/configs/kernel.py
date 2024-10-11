from typing import Literal, Union

from pydantic.v1 import Field, dataclasses

from molflux.modelzoo.models.lightning.config import ConfigDict
from molflux.modelzoo.models.lightning_gp.configs.constraint import (
    ConstraintConfigT,
    PositiveConstraintConfig,
)


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class KernelConfig:
    name: str
    ard_num_dims: int | None = None
    lengthscale_constraint: ConstraintConfigT = Field(
        default_factory=PositiveConstraintConfig,
        discriminator="name",
    )
    eps: float = 1e-6
    active_dims: list[int] | None = None
    add_scale: bool = False


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class CosineKernelConfig(KernelConfig):
    name: Literal["CosineKernel"] = "CosineKernel"
    period_length_constraint: ConstraintConfigT = Field(
        default_factory=PositiveConstraintConfig,
        discriminator="name",
    )


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class JaccardKernelConfig(KernelConfig):
    name: Literal["JaccardKernel"] = "JaccardKernel"


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class LinearKernelConfig(KernelConfig):
    name: Literal["LinearKernel"] = "LinearKernel"
    variance_constraint: ConstraintConfigT = Field(
        default_factory=PositiveConstraintConfig,
        discriminator="name",
    )


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class MaternKernelConfig(KernelConfig):
    name: Literal["MaternKernel"] = "MaternKernel"
    nu: float = 2.5


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class PeriodicKernelConfig(KernelConfig):
    name: Literal["PeriodicKernel"] = "PeriodicKernel"
    period_length_constraint: ConstraintConfigT = Field(
        default_factory=PositiveConstraintConfig,
        discriminator="name",
    )


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class PiecewisePolynomialKernelConfig(KernelConfig):
    name: Literal["PiecewisePolynomialKernel"] = "PiecewisePolynomialKernel"
    q: Literal[0, 1, 2, 3] = 2


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class PolynomialKernelConfig(KernelConfig):
    name: Literal["PolynomialKernel"] = "PolynomialKernel"
    offset_constraint: ConstraintConfigT = Field(
        default_factory=PositiveConstraintConfig,
        discriminator="name",
    )
    power: int = 2


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class RBFKernelConfig(KernelConfig):
    name: Literal["RBFKernel"] = "RBFKernel"


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class RQKernelConfig(KernelConfig):
    name: Literal["RQKernel"] = "RQKernel"
    alpha_constraint: ConstraintConfigT = Field(
        default_factory=PositiveConstraintConfig,
        discriminator="name",
    )


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class AdditiveKernelConfig(KernelConfig):
    kernels: list["KernelConfigT"]
    name: Literal["AdditiveKernel"] = "AdditiveKernel"


@dataclasses.dataclass(config=ConfigDict, kw_only=True)
class ProductKernelConfig(KernelConfig):
    kernels: list["KernelConfigT"]
    name: Literal["ProductKernel"] = "ProductKernel"


KernelConfigT = Union[
    CosineKernelConfig,
    JaccardKernelConfig,
    LinearKernelConfig,
    MaternKernelConfig,
    PeriodicKernelConfig,
    PiecewisePolynomialKernelConfig,
    PolynomialKernelConfig,
    RBFKernelConfig,
    RQKernelConfig,
    AdditiveKernelConfig,
    ProductKernelConfig,
]

AdditiveKernelConfig.__pydantic_model__.update_forward_refs()  # type: ignore[attr-defined]
ProductKernelConfig.__pydantic_model__.update_forward_refs()  # type: ignore[attr-defined]
