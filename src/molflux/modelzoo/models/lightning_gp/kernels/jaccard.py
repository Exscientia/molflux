from typing import Any

try:
    import torch
    from gpytorch.kernels import Kernel

except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("lightning_gp", e) from e


class JaccardKernel(Kernel):
    """
    Computes a covariance matrix based on the Jaccard kernel between inputs.
    Assumes that the input tensor is a binary bit vector.
    """

    has_lengthscale = True

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> torch.Tensor:
        x1_ = x1
        x2_ = x2

        # if the input is 2D, make it 3D. This is done for allowing broadcasting
        # for the divisor computation
        if x1_.dim() == 2:
            x1_ = x1_.unsqueeze(0)
            x2_ = x2_.unsqueeze(0)
            must_squeeze = True
        else:
            must_squeeze = False

        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)
            x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

        prod = torch.matmul(x1_, x2_.transpose(-2, -1))

        divisor = (
            torch.sum(x1_, dim=-1).unsqueeze(-1)
            + torch.sum(x2_, dim=-1).unsqueeze(1)
            - prod
        )

        result: torch.Tensor = (prod / divisor) * self.lengthscale
        if must_squeeze:
            result = result.squeeze(0)
        if diag:
            result = result.diag()

        return result
