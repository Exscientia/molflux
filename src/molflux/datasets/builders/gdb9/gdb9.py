import os
from dataclasses import dataclass
from typing import Any, ClassVar, List, Literal, Optional

import pandas as pd

import datasets
from molflux.datasets.typing import ExamplesGenerator

_BASE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"

_HOMEPAGE = "https://moleculenet.org/"

_DESCRIPTION = """
    QM9 dataset of quantum properties of small molecules with up to 29 atoms and up to 9 heavy atoms.
    """


@dataclass
class GDB9Config(datasets.BuilderConfig):
    backend: Literal["openeye", "rdkit"] = "rdkit"


class GDB9(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = GDB9Config
    config: GDB9Config

    BUILDER_CONFIGS: ClassVar[List[datasets.BuilderConfig]] = [
        GDB9Config(
            name="openeye",
            backend="openeye",
        ),
        GDB9Config(
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
                    "mol_id": datasets.Value("string"),
                    "A": datasets.Value("float"),
                    "B": datasets.Value("float"),
                    "C": datasets.Value("float"),
                    "mu": datasets.Value("float"),
                    "alpha": datasets.Value("float"),
                    "homo": datasets.Value("float"),
                    "lumo": datasets.Value("float"),
                    "gap": datasets.Value("float"),
                    "r2": datasets.Value("float"),
                    "zpve": datasets.Value("float"),
                    "u0": datasets.Value("float"),
                    "u298": datasets.Value("float"),
                    "h298": datasets.Value("float"),
                    "g298": datasets.Value("float"),
                    "cv": datasets.Value("float"),
                    "u0_atom": datasets.Value("float"),
                    "u298_atom": datasets.Value("float"),
                    "h298_atom": datasets.Value("float"),
                    "g298_atom": datasets.Value("float"),
                },
            ),
            homepage=_HOMEPAGE,
        )

    def _split_generators(
        self,
        dl_manager: datasets.DownloadManager,
    ) -> List[datasets.SplitGenerator]:
        archive_path = dl_manager.download_and_extract(_BASE_URL)

        mols_file = os.path.join(archive_path, "gdb9.sdf")
        properties_file = os.path.join(archive_path, "gdb9.sdf.csv")

        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "mols_filepath": mols_file,
                    "properties_filepath": properties_file,
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
        mols_filepath: Optional[str] = None,
        properties_filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> ExamplesGenerator:
        try:
            from openeye.oechem import OEIsReadable, OEWriteMolToBytes, oemolistream
        except ImportError as e:
            from molflux.datasets.exceptions import ExtrasDependencyImportError

            raise ExtrasDependencyImportError("openeye-toolkits", e) from e

        if not OEIsReadable(mols_filepath):
            raise ValueError(f"{mols_filepath} is not a valid file.")

        mols_iterator = oemolistream(mols_filepath).GetOEGraphMols()

        df = pd.read_csv(properties_filepath)

        for mol, (index, labels) in zip(mols_iterator, df.iterrows()):
            example_dict = {
                "mol_bytes": OEWriteMolToBytes(".oeb", mol),
                **{k.lstrip(" "): v for k, v in labels.to_dict().items()},
            }
            yield index, example_dict

    def _generate_examples_rdkit(
        self,
        mols_filepath: Optional[str] = None,
        properties_filepath: Optional[str] = None,
        **kwargs: Any,
    ) -> ExamplesGenerator:
        try:
            from rdkit import Chem
        except ImportError as e:
            from molflux.datasets.exceptions import ExtrasDependencyImportError

            raise ExtrasDependencyImportError("rdkit", e) from e

        supplier = Chem.SDMolSupplier(
            mols_filepath,
            sanitize=False,
            removeHs=False,
        )

        df = pd.read_csv(properties_filepath)

        for mol, (index, labels) in zip(supplier, df.iterrows()):
            example_dict = {
                "mol_bytes": mol.ToBinary(),
                **{k.lstrip(" "): v for k, v in labels.to_dict().items()},
            }
            yield index, example_dict
