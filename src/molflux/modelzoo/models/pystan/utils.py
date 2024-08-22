from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    try:
        import stan
    except ImportError as e:
        from molflux.modelzoo.errors import ExtrasDependencyImportError

        raise ExtrasDependencyImportError("pystan", e) from None


class StanWrapper:
    def __init__(self, filename: str) -> None:
        path_to_stan_code = Path(__file__).parent / filename
        self.model_code: str = path_to_stan_code.read_text()
        self.fit: stan.fit.Fit | None = None  # to store posterior samples
