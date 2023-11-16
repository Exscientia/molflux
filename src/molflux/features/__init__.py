from molflux.features.catalogue import (
    fill_catalogue,
    list_representations,
    register_representation,
)
from molflux.features.load import (
    load_from_dict,
    load_from_dicts,
    load_from_yaml,
    load_representation,
)
from molflux.features.representation import Representation, Representations

# Register all plugins at package import time (to fill catalogue)
fill_catalogue()
