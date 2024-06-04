import os
from pathlib import Path
from typing import Any, ClassVar, List

from tdc.benchmark_group import admet_group

import datasets
from datasets.packaged_modules.csv.csv import Csv, CsvConfig

# from tdc.utils.retrieve_benchmark_names("ADMET_Group")
_ADMET_GROUP_BENCHMARK_NAMES = [
    "caco2_wang",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "bbb_martins",
    "ppbr_az",
    "vdss_lombardo",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "cyp2c9_substrate_carbonmangels",
    "half_life_obach",
    "clearance_microsome_az",
    "clearance_hepatocyte_az",
    "herg",
    "ames",
    "dili",
    "ld50_zhu",
]
_ASSETS_DIR = ".cache/huggingface/admet_benchmark"

_DESCRIPTION = f"""\
TDC ADMET benchmark group.

The following dataset configurations are available:
{_ADMET_GROUP_BENCHMARK_NAMES}
"""

_HOMEPAGE = "https://tdcommons.ai/"

_CITATION = """\
@misc{huang2021therapeutics,
      title={Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development},
      author={Kexin Huang and Tianfan Fu and Wenhao Gao and Yue Zhao and Yusuf Roohani and Jure Leskovec and Connor W. Coley and Cao Xiao and Jimeng Sun and Marinka Zitnik},
      year={2021},
      eprint={2102.09548},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


class ADMETBenchmarks(Csv):
    """

    Examples:
        >>> import molflux.datasets
        >>> molflux.datasets.load_dataset("tdc_admet_benchmarks", config_name="caco2_wang", split=None)  # doctest: +SKIP
        DatasetDict({
            train_validation: Dataset({
                features: ['Drug_ID', 'Drug', 'Y'],
                num_rows: 728
            })
            test: Dataset({
                features: ['Drug_ID', 'Drug', 'Y'],
                num_rows: 182
            })
        })
        >>> molflux.datasets.load_dataset("tdc_admet_benchmarks", config_name="caco2_wang", split="test")  # doctest: +SKIP
        Dataset({
            features: ['Drug_ID', 'Drug', 'Y'],
            num_rows: 182
        })
        >>> molflux.datasets.load_dataset("tdc_admet_benchmarks", config_name="caco2_wang", split="all")  # doctest: +SKIP
        Dataset({
            features: ['Drug_ID', 'Drug', 'Y'],
            num_rows: 910
        })
    """

    BUILDER_CONFIGS: ClassVar[List[datasets.BuilderConfig]] = [
        CsvConfig(name=name) for name in _ADMET_GROUP_BENCHMARK_NAMES
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _split_generators(
        self,
        dl_manager: datasets.DownloadManager,
    ) -> List[datasets.SplitGenerator]:
        # Link up to the appropriate assets subdirectory
        cache_path = Path.home() / Path(_ASSETS_DIR)
        cache_path.mkdir(exist_ok=True)
        admet_group(path=cache_path)

        data_path = os.path.join(str(cache_path / "admet_group"), self.config.name)
        self.config.data_files = {
            "train_validation": os.path.join(data_path, "train_val.csv"),
            "test": os.path.join(data_path, "test.csv"),
        }

        return super()._split_generators(dl_manager=dl_manager)  # type: ignore[no-any-return]
