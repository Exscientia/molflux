from typing import Any, Dict, List

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.rdkit._utils import (
    rdkit_mol_from_smiles,
    to_smiles,
)
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

try:
    from rdkit import Chem

except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None

_TOXICOPHORE_SMARTS = [
    "O=N(~O)a",
    "a[NH2]",
    "a[N;X2]=O",
    "CO[N;X2]=O",
    "N[N;X2]=O",
    "O1[c,C]-[c,C]1",
    "C1NC1",
    "N=[N+]=[N-]",
    "C=[N+]=[N-]",
    "N=N-N",
    "c[N;X2]!@;=[N;X2]c",
    "[OH,NH2][N,O]",
    "[OH]Na",
    "[Cl,Br,I]C",
    "[Cl,Br,I]C=O",
    "[N,S]!@[C;X4]!@[CH2][Cl,Br,I]",
    "[cH]1[cH]ccc2c1c3c(cc2)cc[cH][cH]3",
    "[cH]1cccc2c1[cH][cH]c3c2ccc[cH]3",
    "[$([C,c]OS(=O)(=O)O!@[c,C]),$([c,C]S(=O)(=O)O!@[c,C])]",
    "O=N(~O)N",
    "[$(O=[CH]C=C),$(O=[CH]C=O)]",
    "[N;v4]#N",
    "O=C1CCO1",
    "[CH]=[CH]O",
    "[NH;!R][NH;R]a",
    "[CH3][NH]a",
    "aN([$([OH]),$(O*=O)])[$([#1]),$(C(=O)[CH3]),$([CH3]),$([OH]),$(O*=O)]",
    "a13~a~a~a~a2~a1~a(~a~a~a~3)~a~a~a~2",
    "a1~a~a~a2~a~1~a~a3~a(~a~2)~a~a~a~3",
    "a1~a~a~a2~a~1~a~a~a3~a~2~a~a~a~3",
    "a1~a~a~a~a2~a~1~a3~a(~a~2)~a~a~a~a~3",
    "a1~a~a~a~a2~a~1~a~a3~a(~a~2)~a~a~a~3",
    "a1~a~a~a~a2~a~1~a~a3~a(~a~2)~a~a~a~a~3",
    "a1~a~a~a~a2~a~1~a~a~a3~a~2~a~a~a~3",
    "a1~a~a~a~a2~a~1~a~a~a3~a~2~a~a~a~a~3",
    "a13~a~a~a~a2~a1~a(~a~a~a~3)~a~a~2",
]

_ALERTS = [Chem.MolFromSmarts(s) for s in _TOXICOPHORE_SMARTS]

_DESCRIPTION = """
Substructure matching against toxicophore SMARTS patterns used in
Pires et al (2015) J. Med. Chem. 2015, 58, 9, 4066-4072
"""


class Toxicophores(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        **kwargs: Any,
    ) -> Dict[str, List[List[Any]]]:
        """
        Calculates toxicophore substructure matches.

        Args:
            samples: The molecules for which to calculate descriptors.

        Returns:
            Boolean vector corresponding to absence/presence of substruct
            match for each pattern.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation("toxicophores")
            >>> samples = ["C1NC1"]
            >>> representation.featurise(samples)
            {'toxicophores': [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
        """
        match_list: List[List] = []

        for sample in samples:
            with featurisation_error_harness(sample):
                smiles = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smiles)
                match = [int(mol.HasSubstructMatch(alert)) for alert in _ALERTS]
                match_list.append(match)

        return {self.tag: match_list}
