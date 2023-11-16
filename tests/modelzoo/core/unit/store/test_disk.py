import os.path

import pytest

from molflux.modelzoo.errors import NotTrainedError
from molflux.modelzoo.protocols import Estimator
from molflux.modelzoo.store.io import load_from_store, save_to_store


def test_cannot_save_untrained_model_to_disk(tmp_path, fixture_mock_model_untrained):
    """That cannot save an untrained model to disk."""
    model = fixture_mock_model_untrained

    with pytest.raises(NotTrainedError):
        save_to_store(key=tmp_path, model=model)


def test_model_is_saved_to_disk(tmp_path, fixture_mock_model_trained):
    """That can save a trained model to disk"""
    model = fixture_mock_model_trained
    save_to_store(key=tmp_path, model=model)
    assert True


def test_creates_config_json_file(tmp_path, fixture_mock_model_trained):
    """That a config.json file is created on save"""
    model = fixture_mock_model_trained
    save_to_store(key=tmp_path, model=model)
    expected_config_file = tmp_path / "model_config.json"
    assert expected_config_file.is_file()


def test_creates_model_artefacts_dir(tmp_path, fixture_mock_model_trained):
    """That a model_artefacts/ directory is created on save"""
    model = fixture_mock_model_trained
    save_to_store(key=tmp_path, model=model)
    expected_artefacts_dir = tmp_path / "model_artefacts"
    assert expected_artefacts_dir.is_dir()


def test_loaded_from_disk(tmp_path, fixture_mock_model_trained):
    """That can load a model from disk"""
    model = fixture_mock_model_trained
    save_to_store(key=tmp_path, model=model)
    loaded_model = load_from_store(key=tmp_path)
    assert isinstance(loaded_model, Estimator)


def test_loaded_model_is_invariant(tmp_path, fixture_mock_model_trained):
    """That a model loaded from store encodes all the original information."""
    model = fixture_mock_model_trained
    save_to_store(key=tmp_path, model=model)
    loaded_model = load_from_store(key=tmp_path)
    assert loaded_model.name == model.name
    assert loaded_model.tag == model.tag
    assert loaded_model.config == model.config


def test_load_from_non_existent_directory_does_not_create_spurious_artefacts():
    """That attempting to load a model from an inexistent directory does not
    leave behind a partially initialised folder.
    """
    non_existent_dir = "does/not/exist"

    try:
        load_from_store(key=non_existent_dir)
    except FileNotFoundError:
        pass

    assert not os.path.exists(non_existent_dir)
