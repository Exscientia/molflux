if __name__ == "__main__":
    import h5py

    import datasets

    dl_manager = datasets.DownloadManager()

    h5_file_path = dl_manager.download_and_extract(
        "https://github.com/openmm/spice-dataset/releases/download/1.1.1/SPICE.hdf5",
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

    new_h5_file.close()
