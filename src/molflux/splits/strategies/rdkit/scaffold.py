import logging
from typing import Any, Iterator, List, Optional

try:
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
except ImportError as e:
    from molflux.splits.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from e

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable
from molflux.splits.utils import partition

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Class for doing data splits based on the scaffold of small molecules using RDKit.

Group  molecules  based on  the Bemis-Murcko scaffold representation, which identifies rings,
linkers, frameworks (combinations between linkers and rings) and atomic properties  such as
atom type, hibridization and bond order in a dataset of molecules. Then split the groups by
the number of molecules in each group in decreasing order.

IMPORTANT NOTE: If no core can be extracted, a warning will be raised and the SMILES itself is used as a scaffold.

References:
    .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
    1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.
"""


class Scaffold(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION,
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        train_fraction: float = 0.8,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        include_chirality: bool = False,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Args:
            y: List of smiles to be used for scaffold split
            groups (optional): Group labels for the samples used while splitting the dataset.
            train_fraction: The proportion of the dataset to include in the train split.
            validation_fraction: The proportion of the dataset to include in the validation split.
            test_fraction: The proportion of the dataset to include in the test split.
            include_chirality (optional): Whether to include chirality in scaffolds or not.
                Defaults to False.

        Examples:
            >>> from molflux.splits import load_splitting_strategy
            >>> strategy = load_splitting_strategy('scaffold_rdkit')
            >>> dataset = ['CCCC', 'CC', 'c1ccncc1']
            >>> folds = strategy.split(dataset=dataset, y=dataset)
        """
        if y is None:
            raise ValueError("""y parameter should be provided for scaffold splits.""")

        np.testing.assert_almost_equal(
            train_fraction + validation_fraction + test_fraction,
            1.0,
        )

        train_cutoff, validation_cutoff = partition(
            dataset,
            train_fraction,
            validation_fraction,
        )

        scaffold_sets = generate_scaffolds(y=y, include_chirality=include_chirality)

        train_indices: List[int] = []
        validation_indices: List[int] = []
        test_indices: List[int] = []

        for scaffold_set in scaffold_sets:
            if len(train_indices) + len(scaffold_set) > train_cutoff:
                if (
                    len(train_indices) + len(validation_indices) + len(scaffold_set)
                    > validation_cutoff
                ):
                    test_indices += scaffold_set
                else:
                    validation_indices += scaffold_set
            else:
                train_indices += scaffold_set

        yield train_indices, validation_indices, test_indices


def _generate_scaffold(smiles: str, include_chirality: bool) -> str:
    """Compute the Bemis-Murcko scaffold for a SMILES string.

    Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
    They are essentially that part of the molecule consisting of
    rings and the linker atoms between them.

    Args:
        smiles: The SMILES string.
        include_chirality: Whether to include chirality in scaffolds or not.

    Returns:
        The MurckScaffold SMILES from the original SMILES

    References:
        .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
        1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        scaffold: str = MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=include_chirality,
        )
        return scaffold
    except Exception:
        logger.warning(
            f"Could not find scaffold for molecule {smiles}. Will use the smiles itself as a scaffold.",
        )
        return smiles


def generate_scaffolds(y: ArrayLike, include_chirality: bool) -> List[List[int]]:
    """Returns all scaffolds from the SMILES string provided.

    Returns:
        List of indices of each scaffold in y.
    """

    scaffolds = {}
    for ind, smiles in enumerate(y):
        scaffold = _generate_scaffold(
            smiles=smiles,
            include_chirality=include_chirality,
        )

        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(),
            key=lambda x: (len(x[1]), x[1][0]),
            reverse=True,
        )
    ]
    return scaffold_sets
