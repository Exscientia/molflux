if __name__ == "__main__":
    import h5py

    import datasets

    dl_manager = datasets.DownloadManager()

    h5_file_path = dl_manager.download_and_extract(
        "https://figshare.com/ndownloader/files/18112775",
    )
    h5_file = h5py.File(h5_file_path, "r")

    new_h5_file = h5py.File("data.h5", "w")

    for key in list(h5_file.keys())[:5]:
        new_h5_file.create_group(key)
        new_h5_file[key].create_dataset(
            "atomic_numbers",
            data=h5_file[key]["atomic_numbers"],
        )
        new_h5_file[key].create_dataset(
            "ccsd(t)_cbs.energy",
            data=h5_file[key]["ccsd(t)_cbs.energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "coordinates",
            data=h5_file[key]["coordinates"][:2],
        )
        new_h5_file[key].create_dataset(
            "hf_dz.energy",
            data=h5_file[key]["hf_dz.energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "hf_qz.energy",
            data=h5_file[key]["hf_qz.energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "hf_tz.energy",
            data=h5_file[key]["hf_tz.energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "mp2_dz.corr_energy",
            data=h5_file[key]["mp2_dz.corr_energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "mp2_qz.corr_energy",
            data=h5_file[key]["mp2_qz.corr_energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "mp2_tz.corr_energy",
            data=h5_file[key]["mp2_tz.corr_energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "npno_ccsd(t)_dz.corr_energy",
            data=h5_file[key]["npno_ccsd(t)_dz.corr_energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "npno_ccsd(t)_tz.corr_energy",
            data=h5_file[key]["npno_ccsd(t)_tz.corr_energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "tpno_ccsd(t)_dz.corr_energy",
            data=h5_file[key]["tpno_ccsd(t)_dz.corr_energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_dz.cm5_charges",
            data=h5_file[key]["wb97x_dz.cm5_charges"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_dz.dipole",
            data=h5_file[key]["wb97x_dz.dipole"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_dz.energy",
            data=h5_file[key]["wb97x_dz.energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_dz.forces",
            data=h5_file[key]["wb97x_dz.forces"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_dz.hirshfeld_charges",
            data=h5_file[key]["wb97x_dz.hirshfeld_charges"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_dz.quadrupole",
            data=h5_file[key]["wb97x_dz.quadrupole"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_tz.dipole",
            data=h5_file[key]["wb97x_tz.dipole"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_tz.energy",
            data=h5_file[key]["wb97x_tz.energy"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_tz.forces",
            data=h5_file[key]["wb97x_tz.forces"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_tz.mbis_charges",
            data=h5_file[key]["wb97x_tz.mbis_charges"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_tz.mbis_dipoles",
            data=h5_file[key]["wb97x_tz.mbis_dipoles"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_tz.mbis_octupoles",
            data=h5_file[key]["wb97x_tz.mbis_octupoles"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_tz.mbis_quadrupoles",
            data=h5_file[key]["wb97x_tz.mbis_quadrupoles"][:2],
        )
        new_h5_file[key].create_dataset(
            "wb97x_tz.mbis_volumes",
            data=h5_file[key]["wb97x_tz.mbis_volumes"][:2],
        )

    new_h5_file.close()
