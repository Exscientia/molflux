from colorama import Fore, Style


class DuplicateKeyError(KeyError):
    """Raisable when a mapping (dictionary) key is being duplicated."""


class ExtrasDependencyImportError(Exception):
    """Raisable on ImportErrors due to missing package extras."""

    def __init__(self, extras_type: str, nested_error: Exception):
        message = (
            f"\n\n📦 {nested_error}\n\n"
            + f"{Style.DIM}# have you tried running the following?{Style.RESET_ALL}\n"
            + f"$ {Style.BRIGHT + Fore.GREEN}pip install 'molflux[{extras_type}]'{Style.RESET_ALL}"
        )
        super().__init__(message)
