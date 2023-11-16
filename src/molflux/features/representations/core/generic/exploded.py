from typing import Any, Dict, List

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.typing import ArrayLike

_DESCRIPTION = """
A generic representation that splits iterable molfluxs into individual features.
"""


class Exploded(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(self, samples: ArrayLike, **kwargs: Any) -> Dict[str, List[Any]]:
        """Explodes each array-like sample into individual features.

        Args:
            samples: The data to featurise. Should consist of iterable molfluxs.

        Returns:
            A dictionary of exploded molfluxs.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('exploded')
            >>> samples = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
            >>> representation.featurise(samples)
            {'exploded::0': [1, 10, 100], 'exploded::1': [2, 20, 200], 'exploded::2': [3, 30, 300], 'exploded::3': [4, 40, 400]}
        """
        return {
            f"{self.tag}::{i}": value
            for i, value in enumerate(map(list, zip(*samples)))
        }
