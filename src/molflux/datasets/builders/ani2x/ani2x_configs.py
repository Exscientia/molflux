from typing import Dict, Literal

import datasets

URL_DICT: Dict[str, Dict[str, str]] = {
    "wB97X/631Gd": {
        "url": "https://zenodo.org/records/10108942/files/ANI-2x-wB97X-631Gd.tar.gz",
        "path": "final_h5/ANI-2x-wB97X-631Gd.h5",
    },
    "wB97X/def2TZVPP": {
        "url": "https://zenodo.org/records/10108942/files/ANI-2x-wB97X-def2TZVPP.tar.gz",
        "path": "final_h5/ANI-2x-wB97X-def2TZVPP.h5",
    },
    "wB97MD3BJ/def2TZVPP": {
        "url": "https://zenodo.org/records/10108942/files/ANI-2x-wB97MD3BJ-def2TZVPP.tar.gz",
        "path": "final_h5/ANI-2x-wB97MD3BJ-def2TZVPP.h5",
    },
    "wB97MV/def2TZVPP": {
        "url": "https://zenodo.org/records/10108942/files/ANI-2x-wB97MV-def2TZVPP.tar.gz",
        "path": "final_h5/ANI-2x-wB97MV-def2TZVPP.h5",
    },
    "B973c/def2mTZVP": {
        "url": "https://zenodo.org/records/10108942/files/ANI-2x-B973c-def2mTZVP.tar.gz",
        "path": "final_h5/ANI-2x-B973c-def2mTZVP.h5",
    },
}

LEVEL_OF_THEORY = Literal[
    "wB97X/631Gd",
    "wB97X/def2TZVPP",
    "wB97MD3BJ/def2TZVPP",
    "wB97MV/def2TZVPP",
    "B973c/def2mTZVP",
]

FEATURES = {
    "wB97X/631Gd": {
        "mol_bytes": datasets.Value("binary"),
        "energies": datasets.Value("float64"),
        "forces": datasets.Sequence(
            feature=datasets.Sequence(
                feature=datasets.Value(dtype="float64", id=None),
                length=-1,
                id=None,
            ),
            length=-1,
            id=None,
        ),
    },
    "wB97X/def2TZVPP": {
        "mol_bytes": datasets.Value("binary"),
        "energies": datasets.Value("float64"),
        "dipoles": datasets.Sequence(
            datasets.Value("float64"),
            length=3,
        ),
        "mbis_atomic_charges": datasets.Sequence(
            datasets.Value("float64"),
            length=-1,
        ),
        "mbis_atomic_dipole_magnitudes": datasets.Sequence(
            datasets.Value("float64"),
            length=-1,
        ),
        "mbis_atomic_volumes": datasets.Sequence(
            datasets.Value("float64"),
            length=-1,
        ),
        "mbis_atomic_octupole_magnitudes": datasets.Sequence(
            datasets.Value("float64"),
            length=-1,
        ),
        "mbis_atomic_quadrupole_magnitudes": datasets.Sequence(
            datasets.Value("float64"),
            length=-1,
        ),
    },
    "wB97MD3BJ/def2TZVPP": {
        "mol_bytes": datasets.Value("binary"),
        "energies": datasets.Value("float64"),
        "dipoles": datasets.Sequence(
            datasets.Value("float64"),
            length=3,
        ),
        "wB97M_def2-TZVPP.scf-energies": datasets.Value("float64"),
        "D3.energy-corrections": datasets.Value("float64"),
    },
    "wB97MV/def2TZVPP": {
        "mol_bytes": datasets.Value("binary"),
        "energies": datasets.Value("float64"),
        "dipoles": datasets.Sequence(
            datasets.Value("float64"),
            length=3,
        ),
        "wB97M_def2-TZVPP.scf-energies": datasets.Value("float64"),
        "VV10.energy-corrections": datasets.Value("float64"),
    },
    "B973c/def2mTZVP": {
        "mol_bytes": datasets.Value("binary"),
        "energies": datasets.Value("float64"),
        "forces": datasets.Sequence(
            feature=datasets.Sequence(
                feature=datasets.Value(dtype="float64", id=None),
                length=-1,
                id=None,
            ),
            length=-1,
            id=None,
        ),
        "dipole": datasets.Sequence(
            datasets.Value("float64"),
            length=3,
        ),
        "D3.energy-corrections": datasets.Value("float64"),
        "D3.force-corrections": datasets.Sequence(
            feature=datasets.Sequence(
                feature=datasets.Value(dtype="float64", id=None),
                length=-1,
                id=None,
            ),
            length=-1,
            id=None,
        ),
    },
}
