import logging
from typing import Any, Iterator, List, Optional

import numpy as np
from openeye import oechem, oemedchem

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable
from molflux.splits.utils import partition

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Class for doing data splits based on the scaffold of small molecules using OpenEye.

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
        adjust_h_count: bool = True,
        r_group: bool = False,
        include_unsaturated_heterobonds: bool = True,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Args:
            y: The list of smiles to use for the scaffold split
            train_fraction: The proportion of the dataset to include in the train split.
            validation_fraction: The proportion of the dataset to include in the validation split.
            test_fraction: The proportion of the dataset to include in the test split.
            adjust_h_count: Adjust hydrogen count of the framework.
            r_group: Add R-groups as replacement for bonds broken during fragmentation.
            include_unsaturated_heterobonds: Include unsaturated heterobonds in frameworks (eg exocyclic carbonyls).

        Examples:
            >>> from molflux.splits import load_splitting_strategy
            >>> strategy = load_splitting_strategy('scaffold')
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

        scaffold_sets = generate_scaffolds(
            y=y,
            adjust_h_count=adjust_h_count,
            r_group=r_group,
            include_unsaturated_heterobonds=include_unsaturated_heterobonds,
        )

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


def _generate_scaffold(
    smiles: str,
    adjust_h_count: bool = True,
    r_group: bool = False,
    include_unsaturated_heterobonds: bool = True,
) -> str:
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, smiles)

    options = oemedchem.OEBemisMurckoOptions()

    #  SetRegionTyoe expects a string, rather than an Enum
    options.SetRegionType("Framework")

    if include_unsaturated_heterobonds:
        options.SetUnsaturatedHeteroBonds(True)

    absets = oemedchem.OEGetBemisMurcko(mol, options)
    # needed because absets is not an iterator, could be empty and not return anything or iterate in a for loop
    i = 0
    for abset in absets:  # noqa: B007
        i += 1

    if i > 0:
        framework = oechem.OEGraphMol()
        oechem.OESubsetMol(framework, mol, abset, adjust_h_count, r_group)
        scaffold_smiles: str = oechem.OEMolToSmiles(framework)
        return scaffold_smiles
    else:
        logger.warning(
            f"Could not find scaffold for molecule {smiles}. Will use the smiles itself as a scaffold.",
        )
        return smiles


def generate_scaffolds(
    y: ArrayLike,
    adjust_h_count: bool = True,
    r_group: bool = False,
    include_unsaturated_heterobonds: bool = True,
) -> List[List[int]]:
    """Returns all scaffolds from the SMILES string provided.

    Returns:
        List of indices of each scaffold in y.
    """

    scaffolds = {}
    for ind, smiles in enumerate(y):
        scaffold = _generate_scaffold(
            smiles=smiles,
            adjust_h_count=adjust_h_count,
            r_group=r_group,
            include_unsaturated_heterobonds=include_unsaturated_heterobonds,
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
