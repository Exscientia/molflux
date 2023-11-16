from molflux.features.errors import FeaturisationError


def test_featurisation_error_is_runtime_error():
    """That our custom FeaturisationErrors inherit from RuntimeError.

    This helps narrow the scope of the exception, and allow to except it
    even in libraries not directly importing the package.
    """
    assert issubclass(FeaturisationError, RuntimeError)
