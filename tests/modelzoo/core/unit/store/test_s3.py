import pytest

from molflux.modelzoo.errors import NotTrainedError
from molflux.modelzoo.protocols import Estimator
from molflux.modelzoo.store.io import load_from_store, save_to_store


def test_cannot_save_untrained_model_to_s3(
    fixture_mock_model_untrained,
    fixture_empty_bucket,
):
    """That cannot save an untrained model to s3."""
    model = fixture_mock_model_untrained
    bucket_path = fixture_empty_bucket

    with pytest.raises(NotTrainedError):
        save_to_store(bucket_path, model=model)


def test_model_is_saved_to_s3(fixture_mock_model_trained, fixture_empty_bucket):
    """That can save a trained model to s3"""
    model = fixture_mock_model_trained
    bucket_path = fixture_empty_bucket
    save_to_store(bucket_path, model=model)
    assert True


def test_creates_config_json_file(fixture_mock_model_trained, fixture_empty_bucket):
    """That a config.json file is created on save"""
    model = fixture_mock_model_trained
    bucket_path = fixture_empty_bucket
    save_to_store(bucket_path, model=model)
    expected_config_file = bucket_path / "model_config.json"
    assert expected_config_file.exists()
    assert expected_config_file.is_file()


def test_creates_model_artefacts_dir(fixture_mock_model_trained, fixture_empty_bucket):
    """That a model_artefacts/ sub-directory is created on save"""
    model = fixture_mock_model_trained
    bucket_path = fixture_empty_bucket
    save_to_store(bucket_path, model=model)
    expected_artefacts_dir = bucket_path / "model_artefacts"
    assert expected_artefacts_dir.exists()
    assert expected_artefacts_dir.is_dir()


def test_loaded_from_s3(fixture_mock_model_trained, fixture_empty_bucket):
    """That can load a model from s3"""
    model = fixture_mock_model_trained
    bucket_path = fixture_empty_bucket
    save_to_store(bucket_path, model=model)
    loaded_model = load_from_store(bucket_path)
    assert isinstance(loaded_model, Estimator)


def test_loaded_model_is_invariant(fixture_mock_model_trained, fixture_empty_bucket):
    """That a model loaded from store encodes all the original information."""
    model = fixture_mock_model_trained
    bucket_path = fixture_empty_bucket
    save_to_store(bucket_path, model=model)
    loaded_model = load_from_store(bucket_path)
    assert loaded_model.name == model.name
    assert loaded_model.tag == model.tag
    assert loaded_model.config == model.config
