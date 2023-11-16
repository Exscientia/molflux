from molflux.datasets.catalogue import fill_catalogue, list_datasets
from molflux.datasets.featurisation import featurise_dataset
from molflux.datasets.io import load_dataset_from_store, save_dataset_to_store
from molflux.datasets.load import (
    load_dataset,
    load_dataset_builder,
    load_from_dict,
    load_from_dicts,
    load_from_yaml,
)
from molflux.datasets.splitting import split_dataset

# Register all plugins at package import time (to fill catalogue)
fill_catalogue()
