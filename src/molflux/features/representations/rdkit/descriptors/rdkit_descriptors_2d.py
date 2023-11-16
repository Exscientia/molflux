from __future__ import annotations

import difflib
import itertools
from typing import TYPE_CHECKING, Any, Literal, get_args

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.rdkit._utils import (
    rdkit_mol_from_smiles,
    to_smiles,
)
from molflux.features.utils import featurisation_error_harness

if TYPE_CHECKING:
    from molflux.features.typing import MolArray

try:
    from rdkit.ML.Descriptors import MoleculeDescriptors

except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None


# from [descriptor[0] for descriptor in rdkit.Chem.Descriptors._descList]:
_Descriptor2D = Literal[
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MinEStateIndex",
    "qed",
    "SPS",
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
    "AvgIpc",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "HallKierAlpha",
    "Ipc",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "PEOE_VSA1",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "TPSA",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState1",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount",
    "MolLogP",
    "MolMR",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
]
_ALL_AVAILABLE_DESCRIPTORS = list(get_args(_Descriptor2D))


_DESCRIPTION = f"""
209 rdkit 2d molecular descriptors.

The following descriptors are available: {_ALL_AVAILABLE_DESCRIPTORS!r}

For further info and references, see
https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors.
"""


class RdkitDescriptors_2d(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        include: list[_Descriptor2D] | None = None,
        exclude: list[_Descriptor2D] | None = None,
        **kwargs: Any,
    ) -> dict[str, list[float]]:
        """Calculates 2D molecular descriptors.

        Args:
            samples: The molecules for which to calculate descriptors.
            include: A list of specific 2D descriptor names to calculate. If
                `None`, all available 2D descriptors are calculated. Defaults to
                `None`.
            exclude: A list of specific 2D descriptor names to not calculate. If
                `None`, no descriptors are excluded. Defaults to `None`.

        Returns:
            Calculated values of the 2D descriptor, each descriptor as its own
             feature.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('rdkit_descriptors_2d')
            >>> samples = ['COc1cc2c(cc1OCCCN3CCOCC3)c(ncn2)Sc4nccs4', 'c1ccccc1']
            >>> representation.featurise(samples, include=["SlogP_VSA8", "VSA_EState8"])
            {'rdkit_descriptors_2d::SlogP_VSA8': [10.90..., 0.0], 'rdkit_descriptors_2d::VSA_EState8': [5.25..., 0.0]}
        """

        # not using set difference to avoid possible internal shuffling
        descriptors_to_calculate = [
            x
            for x in (include or _ALL_AVAILABLE_DESCRIPTORS)
            if x not in (exclude or [])
        ]

        if not descriptors_to_calculate:
            raise ValueError(
                "No descriptors to calculate: please expand your 'select' filter and / or reduce your 'exclude' filter.",
            )

        invalid_descriptors = set(descriptors_to_calculate).difference(
            _ALL_AVAILABLE_DESCRIPTORS,
        )
        if invalid_descriptors:
            msg = "The following descriptor(s) are not available:"
            for invalid_descriptor in invalid_descriptors:
                msg += f"\n\t{invalid_descriptor!r}"
                similar = difflib.get_close_matches(
                    invalid_descriptor,
                    _ALL_AVAILABLE_DESCRIPTORS,
                )
                if similar:
                    msg += f" -> You might be looking for one of these: {similar}"
            raise ValueError(msg)

        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            descriptors_to_calculate,
        )

        descriptors_results_list: list[list[float]] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smiles = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smiles)
                descriptors_results_list.append(calculator.CalcDescriptors(mol))

        # Transpose the results and return one feature for each descriptor
        return {  # type: ignore[var-annotated]
            f"{self.tag}::{name}": v
            for name, v in itertools.zip_longest(
                calculator.descriptorNames,
                map(list, zip(*descriptors_results_list)),
                fillvalue=[],
            )
        }


def list_available_rdkit_descriptors_2d() -> list[str]:
    """Returns all available 2D descriptors names.

    This is a convenience function to abstract away the rdkit backend details.
    """
    return _ALL_AVAILABLE_DESCRIPTORS
