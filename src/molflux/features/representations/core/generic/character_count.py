from typing import Any

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.typing import SmilesArray
from molflux.features.utils import assert_n_positional_args

_DESCRIPTION = """
A generic representation that returns the character count of a string
"""


class CharacterCount(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        *columns: SmilesArray,
        without_hs: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[Any]]:
        """Counts the charactes in each string sample.

        Args:
            samples: The data to featurise. Should consist of strings

        Returns:
            The string lengths

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('character_count')
            >>> samples = ["hi", "ho", "hum"]
            >>> representation.featurise(samples)
            {'character_count': [2, 2, 3]}
        """
        assert_n_positional_args(*columns, expected_size=1)
        samples = columns[0]

        if without_hs:
            samples = [sample.replace("H", "").replace("h", "") for sample in samples]

        counts = [len(sample) for sample in samples]

        return {f"{self.tag}": counts}
