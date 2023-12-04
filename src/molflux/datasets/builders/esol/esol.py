from typing import Any, List, Optional

import pandas as pd

import datasets
from molflux.datasets.typing import ExamplesGenerator

_BASE_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
)

_HOMEPAGE = "https://pubs.acs.org/doi/10.1021/ci034243x"

_DESCRIPTION = """
"""


class ESOL(datasets.GeneratorBasedBuilder):
    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "smiles": datasets.Value("string"),
                    "log_solubility": datasets.Value("float64"),
                },
            ),
            homepage=_HOMEPAGE,
        )

    def _split_generators(
        self,
        dl_manager: datasets.DownloadManager,
    ) -> List[datasets.SplitGenerator]:
        archive_path = dl_manager.download(_BASE_URL)

        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "csv_path": archive_path,
                },
            ),
        ]

    def _generate_examples(
        self,
        csv_path: Optional[str] = None,
        **kwargs: Any,
    ) -> ExamplesGenerator:
        df = pd.read_csv(csv_path)

        # filter unparsable smiles
        df = df[
            (df["smiles"] != "c1c(OC)c(OC)C2C(=O)OCC2c1")
            * (df["smiles"] != "c1c(O)C2C(=O)C3cc(O)ccC3OC2cc1(OC)")
        ]

        for index, row in df.iterrows():
            row_dict = row.to_dict()
            example_dict = {
                "smiles": row_dict["smiles"],
                "log_solubility": row_dict["measured log solubility in mols per litre"],
            }
            yield index, example_dict
