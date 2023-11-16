import pytest

from molflux.features.catalogue import list_representations
from molflux.features.errors import ExtrasDependencyImportError
from molflux.features.load import load_representation

_PLUGIN_REPRESENTATIONS = {
    group: names for group, names in list_representations().items() if group != "core"
}


@pytest.mark.parametrize(
    "extra,representation_name",
    [
        (group, name)
        for group in _PLUGIN_REPRESENTATIONS
        for name in _PLUGIN_REPRESENTATIONS[group]
    ],
)
def test_missing_extra_dependencies_raise_on_load(extra, representation_name):
    """That if dependencies are missing for representations offered as an extra,
    an error is raised at load time.

    The error should hint at how to install the missing dependencies.
    """

    with pytest.raises(
        ExtrasDependencyImportError,
        match=rf"pip install \'molflux\[{extra}\]\'",
    ):
        load_representation(representation_name)
