if __name__ == "__main__":
    from pathlib import Path

    import h5py

    import datasets
    from molflux.datasets.builders.ani2x.ani2x_configs import FEATURES, URL_DICT

    dl_manager = datasets.DownloadManager()
    Path("final_h5").mkdir()

    for level_of_theory in URL_DICT.keys():
        h5_file_path = dl_manager.download_and_extract(URL_DICT[level_of_theory]["url"])
        h5_file = h5py.File(f"{h5_file_path}/{URL_DICT[level_of_theory]['path']}", "r")

        new_h5_file = h5py.File(URL_DICT[level_of_theory]["path"], "w")
        for key in list(h5_file.keys())[:5]:
            new_h5_file.create_group(key)
            for feature in FEATURES[level_of_theory]:
                if feature != "mol_bytes":
                    new_h5_file[key].create_dataset(
                        feature,
                        data=h5_file[key][feature][:2],
                    )
            new_h5_file[key].create_dataset(
                "species",
                data=h5_file[key]["species"][:2],
            )
            new_h5_file[key].create_dataset(
                "coordinates",
                data=h5_file[key]["coordinates"][:2],
            )

        new_h5_file.close()
