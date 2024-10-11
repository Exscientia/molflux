from typing import cast

import gpytorch
import torch
from gpytorch.constraints import GreaterThan, Interval, LessThan, Positive
from gpytorch.kernels import (
    AdditiveKernel,
    CosineKernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PiecewisePolynomialKernel,
    PolynomialKernel,
    ProductKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)

from molflux.modelzoo.models.lightning_gp.configs.backbone import (
    BackboneConfigT,
    NNBackboneConfig,
)
from molflux.modelzoo.models.lightning_gp.configs.constraint import (
    ConstraintConfigT,
    GreaterThanConstraintConfig,
    IntervalConstraintConfig,
    LessThanConstraintConfig,
)
from molflux.modelzoo.models.lightning_gp.configs.kernel import (
    AdditiveKernelConfig,
    CosineKernelConfig,
    JaccardKernelConfig,
    KernelConfig,
    LinearKernelConfig,
    MaternKernelConfig,
    PeriodicKernelConfig,
    PiecewisePolynomialKernelConfig,
    PolynomialKernelConfig,
    ProductKernelConfig,
    RBFKernelConfig,
    RQKernelConfig,
)
from molflux.modelzoo.models.lightning_gp.configs.mean import MeanConfig
from molflux.modelzoo.models.lightning_gp.gp_config import GPConfig
from molflux.modelzoo.models.lightning_gp.kernels.jaccard import JaccardKernel


def generate_backbone(backbone_config: BackboneConfigT) -> torch.nn.Module | None:
    name = backbone_config.name
    if name == "NoBackbone":
        return None

    if name == "NNBackbone":
        backbone_config = cast(NNBackboneConfig, backbone_config)

        layers = []
        for layer_config in backbone_config.layer_configs:
            layer_name = layer_config.name
            layer_parameters = layer_config.parameters
            layer_class = getattr(torch.nn, layer_name)
            layer = layer_class(**layer_parameters)
            layers.append(layer)

        return torch.nn.Sequential(*layers)

    else:
        raise RuntimeError(f"Backbone config name {name} not recognised!")


def generate_mean(mean_config: MeanConfig) -> gpytorch.means.Mean:
    fixed = mean_config.fixed
    constant_value = mean_config.constant_value

    mean = gpytorch.means.ConstantMean()
    if constant_value is not None:
        mean.constant.data = torch.tensor(constant_value, dtype=torch.float)
    mean.constant.requires_grad = not fixed

    return mean


def generate_constraint(constraint_config: ConstraintConfigT | None) -> Interval | None:
    if constraint_config is None:
        return None

    if constraint_config.name == "Interval":
        constraint_config = cast(IntervalConstraintConfig, constraint_config)  # type: ignore[redundant-cast]
        constraint = Interval(
            lower_bound=constraint_config.lower_bound,
            upper_bound=constraint_config.upper_bound,
        )

    elif constraint_config.name == "Positive":
        constraint = Positive()

    elif constraint_config.name == "GreaterThan":
        constraint_config = cast(GreaterThanConstraintConfig, constraint_config)  # type: ignore[redundant-cast]
        constraint = GreaterThan(lower_bound=constraint_config.lower_bound)

    elif constraint_config.name == "LessThan":
        constraint_config = cast(LessThanConstraintConfig, constraint_config)  # type: ignore[redundant-cast]
        constraint = LessThan(upper_bound=constraint_config.upper_bound)

    else:
        raise ValueError(
            f"Unrecognised constraint config name - {constraint_config.name}",
        )

    return constraint


def generate_kernel(kernel_config: KernelConfig) -> gpytorch.kernels.Kernel:
    name = kernel_config.name

    if name == "CosineKernel":
        kernel_config = cast(CosineKernelConfig, kernel_config)
        period_length_constraint = generate_constraint(
            kernel_config.period_length_constraint,
        )
        kernel = CosineKernel(
            period_length_constraint=period_length_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "JaccardKernel":
        kernel_config = cast(JaccardKernelConfig, kernel_config)
        lengthscale_constraint = generate_constraint(
            kernel_config.lengthscale_constraint,
        )
        kernel = JaccardKernel(
            lengthscale_constraint=lengthscale_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "LinearKernel":
        kernel_config = cast(LinearKernelConfig, kernel_config)
        variance_constraint = generate_constraint(kernel_config.variance_constraint)
        kernel = LinearKernel(
            variance_constraint=variance_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "MaternKernel":
        kernel_config = cast(MaternKernelConfig, kernel_config)
        lengthscale_constraint = generate_constraint(
            kernel_config.lengthscale_constraint,
        )
        kernel = MaternKernel(
            nu=kernel_config.nu,
            lengthscale_constraint=lengthscale_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "PeriodicKernel":
        kernel_config = cast(PeriodicKernelConfig, kernel_config)
        period_length_constraint = generate_constraint(
            kernel_config.period_length_constraint,
        )
        lengthscale_constraint = generate_constraint(
            kernel_config.lengthscale_constraint,
        )
        kernel = PeriodicKernel(
            period_length_constraint=period_length_constraint,
            lengthscale_constraint=lengthscale_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "PiecewisePolynomicalKernel":
        kernel_config = cast(PiecewisePolynomialKernelConfig, kernel_config)
        lengthscale_constraint = generate_constraint(
            kernel_config.lengthscale_constraint,
        )
        kernel = PiecewisePolynomialKernel(
            q=kernel_config.q,
            lengthscale_constraint=lengthscale_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "PolynomicalKernel":
        kernel_config = cast(PolynomialKernelConfig, kernel_config)
        offset_constraint = generate_constraint(kernel_config.offset_constraint)
        kernel = PolynomialKernel(
            power=kernel_config.power,
            offset_constraint=offset_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "RBFKernel":
        kernel_config = cast(RBFKernelConfig, kernel_config)
        lengthscale_constraint = generate_constraint(
            kernel_config.lengthscale_constraint,
        )
        kernel = RBFKernel(
            lengthscale_constraint=lengthscale_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "RQKernel":
        kernel_config = cast(RQKernelConfig, kernel_config)
        lengthscale_constraint = generate_constraint(
            kernel_config.lengthscale_constraint,
        )
        alpha_constraint = generate_constraint(kernel_config.alpha_constraint)
        kernel = RQKernel(
            lengthscale_constraint=lengthscale_constraint,
            alpha_constraint=alpha_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "JaccardKernel":
        kernel_config = cast(JaccardKernelConfig, kernel_config)
        lengthscale_constraint = generate_constraint(
            kernel_config.lengthscale_constraint,
        )
        kernel = JaccardKernel(
            lengthscale_constraint=lengthscale_constraint,
            ard_num_dims=kernel_config.ard_num_dims,
            active_dims=kernel_config.active_dims,
            eps=kernel_config.eps,
        )
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "AdditiveKernel":
        kernel_config = cast(AdditiveKernelConfig, kernel_config)

        kernels = [generate_kernel(config) for config in kernel_config.kernels]

        kernel = AdditiveKernel(*kernels)
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    elif name == "ProductKernel":
        kernel_config = cast(ProductKernelConfig, kernel_config)

        kernels = [generate_kernel(config) for config in kernel_config.kernels]

        kernel = ProductKernel(*kernels)
        if kernel_config.add_scale:
            kernel = ScaleKernel(kernel)

    else:
        raise ValueError(f"Kernel with {name=} not supported!")

    return kernel


def generate_likelihood(model_config: GPConfig) -> gpytorch.likelihoods.Likelihood:
    noise_constraint_config = model_config.likelihood_config.noise_constraint
    constraint = generate_constraint(noise_constraint_config)

    return gpytorch.likelihoods.GaussianLikelihood(noise_constraint=constraint)
