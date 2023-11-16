from typing import Optional

from colorama import Fore, Style


class ExtrasDependencyImportError(Exception):
    """Raisable on ImportErrors due to missing package extras."""

    def __init__(self, extras_type: str, nested_error: Exception):
        message = (
            f"\n\n📦 {nested_error}\n\n"
            + f"{Style.DIM}# have you tried running the following?{Style.RESET_ALL}\n"
            + f"$ {Style.BRIGHT + Fore.GREEN}pip install 'molflux[{extras_type}]'{Style.RESET_ALL}"
        )

        super().__init__(message)


class OptionalDependencyImportError(Exception):
    """Raisable on ImportErrors due to missing optional dependencies."""

    def __init__(self, dependency: str, package: str):
        message = (
            f"Optional dependency {dependency} missing."
            + f"{Style.DIM}# have you tried running the following?{Style.RESET_ALL}\n"
            + f"$ {Style.BRIGHT + Fore.GREEN}pip install '{package}'{Style.RESET_ALL}"
        )

        super().__init__(message)


class NotTrainedError(ValueError, AttributeError):
    """Exception class to raise if a model is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling.

    References:
        https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.NotFittedError.html
    """

    def __init__(self, message: Optional[str] = None) -> None:
        default_message = "This estimator has not been trained yet: please train the estimator with appropriate arguments."
        super().__init__(message or default_message)
