from molflux.metrics.catalogue import fill_catalogue, list_metrics, register_metric
from molflux.metrics.load import (
    load_from_dict,
    load_from_dicts,
    load_from_yaml,
    load_metric,
    load_metrics,
    load_suite,
)
from molflux.metrics.metric import Metric, Metrics
from molflux.metrics.protocols import supports_prediction_intervals
from molflux.metrics.suites.catalogue import list_suites
from molflux.metrics.suites.discovery import import_suites

# Register all plugins at package import time (to fill catalogue)
fill_catalogue()

# import all suites at package import time (to fill catalogue)
import_suites()
