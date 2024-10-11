try:
    import gpytorch
    import torch

except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning_gp", e) from e

from molflux.modelzoo.models.lightning_gp.configs.utils import (
    generate_kernel,
    generate_mean,
)
from molflux.modelzoo.models.lightning_gp.gp_config import GPConfig


class GPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        model_config: GPConfig,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.model_config = model_config

        self.mean_module = generate_mean(model_config.mean_config)
        self.covar_module = generate_kernel(model_config.kernel_config)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
