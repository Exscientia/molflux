if __name__ == "__main__":
    import h5py

    import datasets

    dl_manager = datasets.DownloadManager()

    h5_file_path = dl_manager.download_and_extract(
        "https://zenodo.org/records/8222043/files/SPICE-1.1.4.hdf5?download=1",
    )

    h5_file = h5py.File(h5_file_path, "r")

    new_h5_file = h5py.File("data.h5", "w")

    for key in list(h5_file.keys())[:5]:
        new_h5_file.create_group(key)
        new_h5_file[key].create_dataset(
            "atomic_numbers",
            data=h5_file[key]["atomic_numbers"],
        )
        new_h5_file[key].create_dataset("smiles", data=h5_file[key]["smiles"])
        new_h5_file[key].create_dataset("subset", data=h5_file[key]["subset"])

        new_h5_file[key].create_dataset(
            "conformations",
            data=h5_file[key]["conformations"][:2],
        )
        new_h5_file[key].create_dataset(
            "dft_total_energy",
            data=h5_file[key]["dft_total_energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "dft_total_gradient",
            data=h5_file[key]["dft_total_gradient"][:2],
        )
        new_h5_file[key].create_dataset(
            "formation_energy",
            data=h5_file[key]["formation_energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "mbis_dipoles",
            data=h5_file[key]["mbis_dipoles"][:2],
        )
        new_h5_file[key].create_dataset(
            "mbis_quadrupoles",
            data=h5_file[key]["mbis_quadrupoles"][:2],
        )
        new_h5_file[key].create_dataset(
            "mbis_octupoles",
            data=h5_file[key]["mbis_octupoles"][:2],
        )
        new_h5_file[key].create_dataset(
            "scf_dipole",
            data=h5_file[key]["scf_dipole"][:2],
        )
        new_h5_file[key].create_dataset(
            "scf_quadrupole",
            data=h5_file[key]["scf_quadrupole"][:2],
        )
        new_h5_file[key].create_dataset(
            "mayer_indices",
            data=h5_file[key]["mayer_indices"][:2],
        )
        new_h5_file[key].create_dataset(
            "wiberg_lowdin_indices",
            data=h5_file[key]["wiberg_lowdin_indices"][:2],
        )
    new_h5_file.close()
