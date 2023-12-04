from dataclasses import dataclass
from typing import Any, ClassVar, List, Literal, Optional

import h5py

import datasets
from molflux.datasets.typing import ExamplesGenerator

_BASE_URL = "https://figshare.com/ndownloader/files/18112775"

_HOMEPAGE = "https://www.nature.com/articles/s41597-020-0473-z"

_DESCRIPTION = """
    The ANI-1x and ANI-1ccx ML-based general-purpose potentials for organic molecules
    were developed through active learning; an automated data diversification process.
    The ANI-1x data set contains multiple QM properties from 5 M density
    functional theory calculations, while the ANI-1ccx data set contains 500 k data
    points obtained with an accurate CCSD(T)/CBS extrapolation. Approximately 14 million
    CPU core-hours were expended to generate this data. Multiple QM calculated properties
    for the chemical molfluxs C, H, N, and O are provided: energies, atomic forces, multipole
    moments, atomic charges, etc.

    WARNING: The molecules here are point clouds. The OpenEye mols do not have any bonds.
    """


@dataclass
class ANI1XConfig(datasets.BuilderConfig):
    backend: Literal["openeye", "rdkit"] = "rdkit"


class ANI1X(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = ANI1XConfig
    config: ANI1XConfig

    BUILDER_CONFIGS: ClassVar[List[datasets.BuilderConfig]] = [
        ANI1XConfig(
            name="openeye",
            backend="openeye",
        ),
        ANI1XConfig(
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
                    "chemical_formula": datasets.Value("string"),
                    "ccsd(t)_cbs.energy": datasets.Value("float64"),
                    "hf_dz.energy": datasets.Value("float64"),
                    "hf_qz.energy": datasets.Value("float64"),
                    "hf_tz.energy": datasets.Value("float64"),
                    "mp2_dz.corr_energy": datasets.Value("float64"),
                    "mp2_qz.corr_energy": datasets.Value("float64"),
                    "mp2_tz.corr_energy": datasets.Value("float64"),
                    "npno_ccsd(t)_dz.corr_energy": datasets.Value("float64"),
                    "npno_ccsd(t)_tz.corr_energy": datasets.Value("float64"),
                    "tpno_ccsd(t)_dz.corr_energy": datasets.Value("float64"),
                    "wb97x_dz.cm5_charges": datasets.Sequence(
                        datasets.Value("float64"),
                    ),
                    "wb97x_dz.dipole": datasets.Sequence(
                        datasets.Value("float64"),
                        length=-1,
                    ),
                    "wb97x_dz.energy": datasets.Value("float64"),
                    "wb97x_dz.forces": datasets.Sequence(
                        feature=datasets.Sequence(
                            feature=datasets.Value(dtype="float64", id=None),
                            length=-1,
                            id=None,
                        ),
                        length=-1,
                        id=None,
                    ),
                    "wb97x_dz.hirshfeld_charges": datasets.Sequence(
                        datasets.Value("float64"),
                    ),
                    "wb97x_dz.quadrupole": datasets.Sequence(
                        datasets.Value("float64"),
                        length=-1,
                    ),
                    "wb97x_tz.dipole": datasets.Sequence(
                        datasets.Value("float64"),
                        length=-1,
                    ),
                    "wb97x_tz.energy": datasets.Value("float64"),
                    "wb97x_tz.forces": datasets.Sequence(
                        feature=datasets.Sequence(
                            feature=datasets.Value(dtype="float64", id=None),
                            length=-1,
                            id=None,
                        ),
                        length=-1,
                        id=None,
                    ),
                    "wb97x_tz.mbis_charges": datasets.Sequence(
                        datasets.Value("float64"),
                    ),
                    "wb97x_tz.mbis_dipoles": datasets.Sequence(
                        datasets.Value("float64"),
                    ),
                    "wb97x_tz.mbis_octupoles": datasets.Sequence(
                        datasets.Value("float64"),
                    ),
                    "wb97x_tz.mbis_quadrupoles": datasets.Sequence(
                        datasets.Value("float64"),
                    ),
                    "wb97x_tz.mbis_volumes": datasets.Sequence(
                        datasets.Value("float64"),
                    ),
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
            from openeye.oechem import OEMol, OEWriteMolToBytes
        except ImportError as e:
            from molflux.datasets.exceptions import ExtrasDependencyImportError

            raise ExtrasDependencyImportError("openeye-toolkits", e) from e

        # open h5 file
        h5_file = h5py.File(h5_file_path, "r+")

        # get columns to loop over
        columns = list(self._info().features.keys())
        columns.remove("mol_bytes")
        columns.append("coordinates")

        # reset index
        index = 0

        # loop over chemical formulae (each have multiple conformers)
        for chemical_formula in h5_file:
            # the dict of data for this chemical formula
            chemical_formula_dict = {
                key: value[:].tolist()
                for key, value in dict(h5_file[chemical_formula]).items()
            }

            # atomic numbers list for the chemical formula
            atomic_nums = chemical_formula_dict["atomic_numbers"]
            del chemical_formula_dict["atomic_numbers"]

            # make all the molecules into mol bytes (only atomic numbers and coords available)
            mol_list = []
            for coords in chemical_formula_dict["coordinates"]:
                mol = OEMol()
                for atom_num in atomic_nums:
                    mol.NewAtom(atom_num)
                flat_coords = [xs for x in coords for xs in x]
                mol.SetCoords(flat_coords)

                mol_bytes = OEWriteMolToBytes(".oeb", mol)
                mol_list.append(mol_bytes)

            # delete coordinates from the dict and add mols
            del chemical_formula_dict["coordinates"]
            chemical_formula_dict["mol_bytes"] = mol_list

            # find list of example dicts (swap dict of lists to list of dicts)
            chemical_formula_examples_list = [
                dict(zip(chemical_formula_dict, t))
                for t in zip(*chemical_formula_dict.values())
            ]

            # yield the examples for the conformers
            for example_dict in chemical_formula_examples_list:
                index += 1

                # add chemical formula
                example_dict["chemical_formula"] = chemical_formula
                yield index, example_dict

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
        columns.append("coordinates")

        # reset index
        index = 0

        # loop over chemical formulae (each have multiple conformers)
        for chemical_formula in h5_file:
            # the dict of data for this chemical formula
            chemical_formula_dict = {
                key: value[:].tolist()
                for key, value in dict(h5_file[chemical_formula]).items()
            }

            # atomic numbers list for the chemical formula
            atomic_nums = chemical_formula_dict["atomic_numbers"]
            del chemical_formula_dict["atomic_numbers"]

            # make all the molecules into mol bytes (only atomic numbers and coords available)
            mol_list = []
            for coords in chemical_formula_dict["coordinates"]:
                mol = RWMol()
                for _idx, atomic_num in enumerate(atomic_nums):
                    atom = Atom(atomic_num)
                    mol.AddAtom(atom)

                conf = Conformer()
                for i in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(i, Point3D(*coords[i]))

                mol.AddConformer(conf)

                mol_list.append(mol.ToBinary())

            # delete coordinates from the dict and add mols
            del chemical_formula_dict["coordinates"]
            chemical_formula_dict["mol_bytes"] = mol_list

            # find list of example dicts (swap dict of lists to list of dicts)
            chemical_formula_examples_list = [
                dict(zip(chemical_formula_dict, t))
                for t in zip(*chemical_formula_dict.values())
            ]

            # yield the examples for the conformers
            for example_dict in chemical_formula_examples_list:
                index += 1

                # add chemical formula
                example_dict["chemical_formula"] = chemical_formula
                yield index, example_dict
