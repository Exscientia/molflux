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
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz",
    )
    mols_file = os.path.join(archive_path, "gdb9.sdf")
    properties_file = os.path.join(archive_path, "gdb9.sdf.csv")

    df = pd.read_csv(properties_file)
    df = df.iloc[:10]

    df.to_csv("gdb9.sdf.csv", index=False)

    mols = oechem.oemolistream(mols_file).GetOEGraphMols()

    mols_10 = [next(mols) for _ in range(10)]

    write_mols(mols_10, "gdb9.sdf")
