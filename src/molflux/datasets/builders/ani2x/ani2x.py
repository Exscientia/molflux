from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Optional

import h5py

import datasets
from molflux.datasets.typing import ExamplesGenerator

from .ani2x_configs import FEATURES, LEVEL_OF_THEORY, URL_DICT

_BASE_URL = "https://zenodo.org/records/10108942"

_HOMEPAGE = "https://pubs.acs.org/doi/10.1021/acs.jctc.0c00121"

_DESCRIPTION = """
    The new model, dubbed ANI-2x, is trained to three additional chemical elements: S, F, and Cl.
    Additionally, ANI-2x underwent torsional refinement training to better predict molecular torsion
    profiles. These new features open a wide range of new applications within organic chemistry and
    drug development. These seven elements (H, C, N, O, F, Cl, and S) make up ~90% of drug-like molecules.

    WARNING: The molecules here are point clouds. The OpenEye mols do not have any bonds.
    """


@dataclass
class ANI2XConfig(datasets.BuilderConfig):
    backend: Literal["openeye", "rdkit"] = "rdkit"
    level_of_theory: LEVEL_OF_THEORY = "wB97X/631Gd"


class ANI2X(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = ANI2XConfig
    config: ANI2XConfig

    BUILDER_CONFIGS: ClassVar[List[datasets.BuilderConfig]] = [
        ANI2XConfig(
            name="rdkit",
            backend="rdkit",
            level_of_theory="wB97X/631Gd",
        ),
        ANI2XConfig(
            name="openeye",
            backend="openeye",
            level_of_theory="wB97X/631Gd",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(FEATURES[self.config.level_of_theory]),
            homepage=_HOMEPAGE,
        )

    def _split_generators(
        self,
        dl_manager: datasets.DownloadManager,
    ) -> List[datasets.SplitGenerator]:
        archive_path = dl_manager.download_and_extract(
            URL_DICT[self.config.level_of_theory]["url"],
        )

        archive_path = str(
            Path(archive_path) / URL_DICT[self.config.level_of_theory]["path"],
        )
        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "h5_file_path": archive_path,
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
        h5_file_path: Optional[str] = None,
        **kwargs: Any,
    ) -> ExamplesGenerator:
        try:
            from openeye.oechem import OEMol, OEWriteMolToBytes
        except ImportError as e:
            from molflux.datasets.exceptions import ExtrasDependencyImportError

            raise ExtrasDependencyImportError("openeye-toolkits", e) from e

        # open h5 file
        h5_file = h5py.File(h5_file_path, "r+")

        # get columns to loop over
        columns = list(self._info().features.keys())
        columns.remove("mol_bytes")

        # reset index
        index = 0
        for _num_atoms, properties in h5_file.items():
            properties_list = [
                dict(zip(properties, t)) for t in zip(*properties.values())
            ]

            for prop in properties_list:
                mol = OEMol()
                for atom_num in prop["species"]:
                    mol.NewAtom(int(atom_num))
                flat_coords = [float(xs) for x in prop["coordinates"] for xs in x]
                mol.SetCoords(flat_coords)

                mol_bytes = OEWriteMolToBytes(".oeb", mol)

                example_dict = {
                    "mol_bytes": mol_bytes,
                }
                for feature in columns:
                    example_dict[feature] = prop[feature]

                index += 1
                yield index - 1, example_dict

    def _generate_examples_rdkit(
        self,
        h5_file_path: Optional[str] = None,
        **kwargs: Any,
    ) -> ExamplesGenerator:
        try:
            from rdkit.Chem import Atom, Conformer, RWMol
            from rdkit.Geometry import Point3D
        except ImportError as e:
            from molflux.datasets.exceptions import ExtrasDependencyImportError

            raise ExtrasDependencyImportError("rdkit", e) from e

        # open h5 file
        h5_file = h5py.File(h5_file_path, "r+")

        # get columns to loop over
        columns = list(self._info().features.keys())
        columns.remove("mol_bytes")

        # reset index
        index = 0
        for _num_atoms, properties in h5_file.items():
            properties_list = [
                dict(zip(properties, t)) for t in zip(*properties.values())
            ]

            for prop in properties_list:
                mol = RWMol()
                for _idx, atomic_num in enumerate(prop["species"]):
                    atom = Atom(int(atomic_num))
                    mol.AddAtom(atom)

                conf = Conformer()
                for i in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, Point3D(*(prop["coordinates"][i].tolist())))

                mol.AddConformer(conf)

                mol_bytes = mol.ToBinary()

                example_dict = {
                    "mol_bytes": mol_bytes,
                }
                for feature in columns:
                    example_dict[feature] = prop[feature]

                index += 1
                yield index - 1, example_dict
