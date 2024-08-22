from typing import Any

from molflux.features.utils import assert_n_positional_args

try:
    from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
    from rdkit.Chem import DataStructs, rdReducedGraphs
    from rdkit.Chem.rdMolDescriptors import GetHashedMorganFingerprint
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

    from molflux.features.representations.rdkit._utils import rdkit_mol_from_smiles
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None

import numpy as np

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.rdkit._utils import to_smiles
from molflux.features.typing import Fingerprint, MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
MapLight features, as detailed in the ADMET property prediction through combinations of molecular fingerprints.

Source code adapted to molflux: https://github.com/maplightrx/MapLight-TDC/blob/main/maplight.py
"""


class MapLight(RepresentationBase):
    OUTPUT_SIZE: int = 2563

    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        *columns: MolArray,
        **kwargs: Any,
    ) -> dict[str, list[Fingerprint]]:
        """Generates the MapLight features for each input molecule.
        This includes 1024 Morgan bits, 1024 Avalon bits, 315 Reduced Graph bits and 200 descriptors, for a total
        of 2563 features.

        Args:
            samples: The molecules to be featurised.

        Returns:
            MapLight features, as lists of floats.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('map_light')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples)  # doctest:+ELLIPSIS
            {'map_light': [[0.0, 0.0, 0.0, 0.0, ...
        """

        assert_n_positional_args(*columns, expected_size=1)
        samples = columns[0]

        morgan_fingerprint_list = []
        avalon_fingerprint_list = []
        erg_fingerprint_list = []
        rdkit_features_list = []

        calculator = MolecularDescriptorCalculator(get_chosen_descriptors())

        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)

                # compute the Morgan Fingerprint
                morgan_fp = GetHashedMorganFingerprint(mol, nBits=1024, radius=2)
                morgan_fp = count_to_array(morgan_fp)
                morgan_fingerprint_list.append(morgan_fp)

                # compute the Avalon Fingerprint
                avalon_fp = GetAvalonCountFP(mol, nBits=1024)
                avalon_fp = count_to_array(avalon_fp)
                avalon_fingerprint_list.append(avalon_fp)

                # compute the Reduced Graph Fingerprint
                erg_fp = rdReducedGraphs.GetErGFingerprint(mol)
                erg_fingerprint_list.append(erg_fp)

                # compute the RDKit descriptors
                rdkit_features = calculator.CalcDescriptors(mol)
                rdkit_features = np.array(rdkit_features)
                rdkit_features_list.append(rdkit_features)

        # stack the features vertically in 4 2-dimensional numpy arrays
        fingerprints = [
            np.vstack(morgan_fingerprint_list),
            np.vstack(avalon_fingerprint_list),
            np.vstack(erg_fingerprint_list),
            np.vstack(rdkit_features_list),
        ]

        # concatenate the features along the column axis
        result = np.concatenate(fingerprints, axis=1)

        return {self.tag: result.tolist()}


def count_to_array(fingerprint: Any) -> np.ndarray:
    array = np.zeros((0,), dtype=np.int8)

    DataStructs.ConvertToNumpyArray(fingerprint, array)

    return array


# from https://www.blopig.com/blog/2022/06/how-to-turn-a-molecule-into-a-vector-of-physicochemical-descriptors-using-rdkit/
def get_chosen_descriptors() -> list[str]:
    """Simple function for returning a list of handcrafted rdkit features"""
    chosen_descriptors = [
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
        "ExactMolWt",
        "FpDensityMorgan1",
        "FpDensityMorgan2",
        "FpDensityMorgan3",
        "FractionCSP3",
        "HallKierAlpha",
        "HeavyAtomCount",
        "HeavyAtomMolWt",
        "Ipc",
        "Kappa1",
        "Kappa2",
        "Kappa3",
        "LabuteASA",
        "MaxAbsEStateIndex",
        "MaxAbsPartialCharge",
        "MaxEStateIndex",
        "MaxPartialCharge",
        "MinAbsEStateIndex",
        "MinAbsPartialCharge",
        "MinEStateIndex",
        "MinPartialCharge",
        "MolLogP",
        "MolMR",
        "MolWt",
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
        "NumRadicalElectrons",
        "NumRotatableBonds",
        "NumSaturatedCarbocycles",
        "NumSaturatedHeterocycles",
        "NumSaturatedRings",
        "NumValenceElectrons",
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
        "RingCount",
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
        "qed",
    ]

    return chosen_descriptors
