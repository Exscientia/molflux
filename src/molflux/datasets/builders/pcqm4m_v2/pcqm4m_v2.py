import os
from dataclasses import dataclass
from typing import Any, ClassVar, List, Literal, Optional

import pandas as pd

import datasets
from molflux.datasets.typing import ExamplesGenerator

_BASE_URL_DICT = {
    "mols_file": "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz",
    "dataset_file": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip",
}

_HOMEPAGE = "https://ogb.stanford.edu/docs/lsc/pcqm4mv2/"

_DESCRIPTION = """
   PCQM4Mv2 is a quantum chemistry dataset originally curated under the PubChemQC project.
   Based on the PubChemQC, we define a meaningful ML task of predicting DFT-calculated HOMO-LUMO
   energy gap of molecules given their 2D molecular graphs. The HOMO-LUMO gap is one of the most
   practically-relevant quantum chemical properties of molecules since it is related to reactivity,
   photoexcitation, and charge transport. Moreover, predicting the quantum chemical property only from
   2D molecular graphs without their 3D equilibrium structures is also practically favorable. This is
   because obtaining 3D equilibrium structures requires DFT-based geometry optimization, which is
   expensive on its own.

   We provide molecules as the SMILES strings, from which 2D molecule graphs (nodes are atoms and edges
   are chemical bonds). We further provide the equilibrium 3D graph structure for training molecules.
    """


@dataclass
class PCQM4MV2Config(datasets.BuilderConfig):
    backend: Literal["openeye", "rdkit"] = "rdkit"


class PCQM4MV2(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = PCQM4MV2Config
    config: PCQM4MV2Config

    BUILDER_CONFIGS: ClassVar[List[datasets.BuilderConfig]] = [
        PCQM4MV2Config(
            name="openeye",
            backend="openeye",
        ),
        PCQM4MV2Config(
            name="rdkit",
            backend="rdkit",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "mol_bytes": datasets.Value("binary"),
                    "smiles": datasets.Value("string"),
                    "homo_lumo_gap": datasets.Value("float64"),
                },
            ),
            homepage=_HOMEPAGE,
        )

    def _split_generators(
        self,
        dl_manager: datasets.DownloadManager,
    ) -> List[datasets.SplitGenerator]:
        raw_file_dict = dl_manager.download_and_extract(_BASE_URL_DICT)

        mols_file = os.path.join(raw_file_dict["mols_file"], "pcqm4m-v2-train.sdf")
        csv_file = dl_manager.extract(
            os.path.join(raw_file_dict["dataset_file"], "pcqm4m-v2/raw/data.csv.gz"),
        )

        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "mols_file": mols_file,
                    "csv_file": csv_file,
                },
            ),
        ]

    def _generate_examples(self, **kwargs: Any) -> ExamplesGenerator:
        if self.config.backend == "openeye":
            return self._generate_examples_openeye(**kwargs)
        elif self.config.backend == "rdkit":
            return self._generate_examples_rdkit(**kwargs)
        else:
            raise KeyError("Unknown backend")

    def _generate_examples_openeye(
        self,
        mols_file: Optional[str] = None,
        csv_file: Optional[str] = None,
        **kwargs: Any,
    ) -> ExamplesGenerator:
        try:
            from openeye.oechem import (
                OEErrorLevel_Error,
                OEThrow,
                OEWriteMolToBytes,
                oemolistream,
            )
        except ImportError as e:
            from molflux.datasets.exceptions import ExtrasDependencyImportError

            raise ExtrasDependencyImportError("openeye-toolkits", e) from e

        OEThrow.SetLevel(OEErrorLevel_Error)

        df = pd.read_csv(csv_file)

        # get rid of points with no homolumogap value
        df = df[~df["homolumogap"].isna()][["smiles", "homolumogap"]]
        df = df.rename(columns={"homolumogap": "homo_lumo_gap"})

        # find mol iterator
        mols_iterator = oemolistream(mols_file).GetOEGraphMols()

        for index, (mol, (_, row)) in enumerate(zip(mols_iterator, df.iterrows())):
            row_dict = row.to_dict()
            row_dict["mol_bytes"] = OEWriteMolToBytes(".oeb", mol)

            yield index, row_dict

    def _generate_examples_rdkit(
        self,
        mols_file: Optional[str] = None,
        csv_file: Optional[str] = None,
        **kwargs: Any,
    ) -> ExamplesGenerator:
        try:
            from rdkit import Chem
        except ImportError as e:
            from molflux.datasets.exceptions import ExtrasDependencyImportError

            raise ExtrasDependencyImportError("rdkit", e) from e

        df = pd.read_csv(csv_file)

        # get rid of points with no homolumogap value
        df = df[~df["homolumogap"].isna()][["smiles", "homolumogap"]]
        df = df.rename(columns={"homolumogap": "homo_lumo_gap"})

        # find mol iterator
        supplier = Chem.SDMolSupplier(
            mols_file,
            sanitize=False,
            removeHs=False,
        )

        for index, (mol, (_, row)) in enumerate(zip(supplier, df.iterrows())):
            row_dict = row.to_dict()
            row_dict["mol_bytes"] = mol.ToBinary()

            yield index, row_dict
