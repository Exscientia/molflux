import molflux.features
from molflux.datasets.interfaces import Representations


def test_integrates_with_exs_prism():
    """That the Representations interface used by datasets is compatible
    with the Representations defined by molflux.features.

    That is, we check that datasets works with Representations from molflux.features.
    """
    assert isinstance(molflux.features.Representations, Representations)
