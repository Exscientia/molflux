from typing import Any

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.typing import ArrayLike

_DESCRIPTION = """
A generic representation that sums iterable elements.
"""


class Sum(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(self, *columns: ArrayLike, **kwargs: Any) -> dict[str, list[Any]]:
        """Sums every input column.

        Args:
            *columns: The data to sum. Should consist of iterable elements.

        Returns:
            A dictionary of the sum of columns.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('sum')
            >>> columns = [1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]
            >>> representation.featurise(*columns)
            {'sum': [111, 222, 333, 444]}
        """

        return {f"{self.tag}": [sum(t) for t in zip(*columns, strict=False)]}
