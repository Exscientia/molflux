if __name__ == "__main__":
    import os

    import pandas as pd
    from openeye import oechem

    import datasets

    def write_mols(mols, output_file: str) -> None:  # type: ignore
        # dest must be a valid output format
        if not oechem.OEIsWriteable(output_file):
            raise ValueError(
                f"`{output_file}` is not a supported chemical structure format.",
            )

        num_mols = 0
        with oechem.oemolostream(output_file) as oss:
            for mol in mols:
                # written as a copy since OEWriteMolecule can change the object
                oechem.OEWriteMolecule(oss, mol.CreateCopy())
                num_mols += 1

    dl_manager = datasets.DownloadManager()

    archive_path = dl_manager.download_and_extract(
        {
            "mols_file": "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz",
            "dataset_file": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip",
        },
    )
    mols_file = os.path.join(archive_path["mols_file"], "pcqm4m-v2-train.sdf")
    properties_file = os.path.join(
        archive_path["dataset_file"],
        "pcqm4m-v2/raw/data.csv.gz",
    )

    df = pd.read_csv(properties_file, compression="gzip")
    df = df.iloc[:10]

    os.makedirs("pcqm4m-v2/raw/")
    df.to_csv("pcqm4m-v2/raw/data.csv.gz", index=False, compression="gzip")

    mols = oechem.oemolistream(mols_file).GetOEGraphMols()

    mols_10 = [next(mols) for _ in range(10)]

    write_mols(mols_10, "pcqm4m-v2-train.sdf")
