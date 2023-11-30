from dataclasses import dataclass
from typing import Any, ClassVar, List, Literal, Optional

import h5py

import datasets
from molflux.datasets.typing import ExamplesGenerator

_BASE_URL = "https://github.com/openmm/spice-dataset/releases/download/1.1.1/SPICE.hdf5"

_HOMEPAGE = "https://github.com/openmm/spice-dataset"

_DESCRIPTION = """
    SPICE dataset, a new quantum chemistry dataset for training potentials relevant to simulating drug-like small
    molecules interacting with proteins. It contains over 1.1 million conformations for a diverse set of small
    molecules, dimers, dipeptides, and solvated amino acids. It includes 15 molfluxs, charged and uncharged molecules,
    and a wide range of covalent and non-covalent interactions. It provides both forces and energies
    calculated at the ωB97M-D3(BJ)/def2-TZVPPD level of theory.

    WARNING: The molecules here are point clouds. The OpenEye mols do not have any bonds. The charges are added to the
    mols from the smiles provided.
    """


@dataclass
class SPICEConfig(datasets.BuilderConfig):
    backend: Literal["openeye", "rdkit"] = "rdkit"


class SPICE(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = SPICEConfig
    config: SPICEConfig

    BUILDER_CONFIGS: ClassVar[List[datasets.BuilderConfig]] = [
        SPICEConfig(
            name="openeye",
            backend="openeye",
        ),
        SPICEConfig(
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
                    "dft_total_energy": datasets.Value("float64"),
                    "dft_total_gradient": datasets.Sequence(
                        feature=datasets.Sequence(
                            feature=datasets.Value(dtype="float64", id=None),
                            length=-1,
                            id=None,
                        ),
                        length=-1,
                        id=None,
                    ),
                    "formation_energy": datasets.Value("float64"),
                    "smiles": datasets.Value("string"),
                    "subset": datasets.Value("string"),
                },
            ),
            homepage=_HOMEPAGE,
        )

    def _split_generators(
        self,
        dl_manager: datasets.DownloadManager,
    ) -> List[datasets.SplitGenerator]:
        archive_path = dl_manager.download_and_extract(_BASE_URL)

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
            from openeye.oechem import OEMol, OESmilesToMol, OEWriteMolToBytes
        except ImportError as e:
            from molflux.datasets.exceptions import ExtrasDependencyImportError

            raise ExtrasDependencyImportError("openeye-toolkits", e) from e

        # open h5 file
        h5_file = h5py.File(h5_file_path, "r+")

        # get columns to loop over
        columns = list(self._info().features.keys())
        columns.remove("mol_bytes")
        columns.append("conformations")

        # reset index
        index = 0

        # loop over chemical formulae (each have multiple conformers)
        for chemical_formula_id in h5_file:
            # the dict of data for this chemical formula
            chemical_formula_dict = {
                key: value[:].tolist()
                for key, value in dict(h5_file[chemical_formula_id]).items()
            }

            # get smiles and subsets for this id
            chemical_formula_id_smiles = chemical_formula_dict["smiles"][0].decode(
                "utf-8",
            )
            chemical_formula_id_subset = chemical_formula_dict["subset"][0].decode(
                "utf-8",
            )
            del chemical_formula_dict["smiles"]
            del chemical_formula_dict["subset"]

            # get charges dictionary from smiles to add to mol
            smiles_mol = OEMol()
            OESmilesToMol(smiles_mol, chemical_formula_id_smiles)
            charge_dict = {}
            for atom in smiles_mol.GetAtoms():
                atom_map_idx = atom.GetMapIdx()
                if atom_map_idx != 0:
                    charge_dict[atom_map_idx - 1] = atom.GetFormalCharge()
                else:
                    raise RuntimeError(
                        "Cannot get atom map idx while extracting charges from smiles.",
                    )

            # atomic numbers list for the chemical formula
            atomic_nums = chemical_formula_dict["atomic_numbers"]
            del chemical_formula_dict["atomic_numbers"]

            # make all the molecules into mol bytes (only atomic numbers and coords available)
            mol_list = []
            for coords in chemical_formula_dict["conformations"]:
                mol = OEMol()
                for idx, atom_num in enumerate(atomic_nums):
                    atom = mol.NewAtom(atom_num)
                    atom.SetFormalCharge(charge_dict[idx])

                flat_coords = [xs for x in coords for xs in x]
                mol.SetCoords(flat_coords)

                mol_bytes = OEWriteMolToBytes(".oeb", mol)
                mol_list.append(mol_bytes)

            # delete coordinates from the dict and add mols
            del chemical_formula_dict["conformations"]
            chemical_formula_dict["mol_bytes"] = mol_list

            # find list of example dicts (swap dict of lists to list of dicts)
            chemical_formula_examples_list = [
                dict(zip(chemical_formula_dict, t))
                for t in zip(*chemical_formula_dict.values())
            ]

            # yield the examples for the conformers
            for example_dict in chemical_formula_examples_list:
                index += 1
                example_dict["smiles"] = chemical_formula_id_smiles
                example_dict["subset"] = chemical_formula_id_subset

                yield index, example_dict

    def _generate_examples_rdkit(
        self,
        h5_file_path: Optional[str] = None,
        **kwargs: Any,
    ) -> ExamplesGenerator:
        try:
            from rdkit.Chem import Atom, Conformer, MolFromSmiles, RWMol
            from rdkit.Geometry import Point3D
        except ImportError as e:
            from molflux.datasets.exceptions import ExtrasDependencyImportError

            raise ExtrasDependencyImportError("rdkit", e) from e

        # open h5 file
        h5_file = h5py.File(h5_file_path, "r+")

        # get columns to loop over
        columns = list(self._info().features.keys())
        columns.remove("mol_bytes")
        columns.append("conformations")

        # reset index
        index = 0

        # loop over chemical formulae (each have multiple conformers)
        for chemical_formula_id in h5_file:
            # the dict of data for this chemical formula
            chemical_formula_dict = {
                key: value[:].tolist()
                for key, value in dict(h5_file[chemical_formula_id]).items()
            }

            # get smiles and subsets for this id
            chemical_formula_id_smiles = chemical_formula_dict["smiles"][0].decode(
                "utf-8",
            )
            chemical_formula_id_subset = chemical_formula_dict["subset"][0].decode(
                "utf-8",
            )
            del chemical_formula_dict["smiles"]
            del chemical_formula_dict["subset"]

            # get charges dictionary from smiles to add to mol
            smiles_mol = MolFromSmiles(chemical_formula_id_smiles, sanitize=False)
            charge_dict = {}
            for atom in smiles_mol.GetAtoms():
                atom_map_idx = atom.GetAtomMapNum()
                if atom_map_idx != 0:
                    charge_dict[atom_map_idx - 1] = atom.GetFormalCharge()
                else:
                    raise RuntimeError(
                        "Cannot get atom map idx while extracting charges from smiles.",
                    )

            # atomic numbers list for the chemical formula
            atomic_nums = chemical_formula_dict["atomic_numbers"]
            del chemical_formula_dict["atomic_numbers"]

            # make all the molecules into mol bytes (only atomic numbers and coords available)
            mol_list = []
            for coords in chemical_formula_dict["conformations"]:
                mol = RWMol()
                for idx, atomic_num in enumerate(atomic_nums):
                    atom = Atom(atomic_num)
                    atom.SetFormalCharge(charge_dict[idx])
                    mol.AddAtom(atom)

                conf = Conformer()
                for i in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, Point3D(*coords[i]))

                mol.AddConformer(conf)
                mol_bytes = mol.ToBinary()
                mol_list.append(mol_bytes)

            # delete coordinates from the dict and add mols
            del chemical_formula_dict["conformations"]
            chemical_formula_dict["mol_bytes"] = mol_list

            # find list of example dicts (swap dict of lists to list of dicts)
            chemical_formula_examples_list = [
                dict(zip(chemical_formula_dict, t))
                for t in zip(*chemical_formula_dict.values())
            ]

            # yield the examples for the conformers
            for example_dict in chemical_formula_examples_list:
                index += 1
                example_dict["smiles"] = chemical_formula_id_smiles
                example_dict["subset"] = chemical_formula_id_subset

                yield index, example_dict
