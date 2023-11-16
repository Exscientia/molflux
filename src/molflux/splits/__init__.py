from molflux.splits.catalogue import (
    fill_catalogue,
    list_splitting_strategies,
    register_splitting_strategy,
)
from molflux.splits.load import (
    load_from_dict,
    load_from_dicts,
    load_from_yaml,
    load_splitting_strategy,
)
from molflux.splits.presets import (
    k_fold_split,
    train_test_split,
    train_validation_test_split,
)
from molflux.splits.strategy import SplittingStrategy

# Register all plugins at package import time (to fill catalogue)
fill_catalogue()
