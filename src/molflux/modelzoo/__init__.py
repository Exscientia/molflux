from molflux.modelzoo.catalogue import fill_catalogue, list_models, register_model
from molflux.modelzoo.load import (
    load_from_dict,
    load_from_dicts,
    load_from_yaml,
    load_model,
)
from molflux.modelzoo.model import ClassificationMixin, ModelBase
from molflux.modelzoo.protocols import (
    Model,
    Models,
    supports_classification,
    supports_covariance,
    supports_prediction_interval,
    supports_sampling,
    supports_std,
    supports_uncertainty_calibration,
)
from molflux.modelzoo.store.io import load_from_store, save_to_store

# Register all plugins at package import time (to fill catalogue)
fill_catalogue()
