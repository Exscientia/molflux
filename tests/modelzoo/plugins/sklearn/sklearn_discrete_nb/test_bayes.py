from __future__ import annotations

import numpy as np
import pytest

from molflux.modelzoo.models.sklearn.sklearn_discrete_nb.bayes import (  # type: ignore
    CorrectedNB,
    CoverageNB,
    PipelinePilotNB,
)


@pytest.fixture
def fixture_training_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.random.choice([0, 1], size=(10, 100))
    y = np.random.choice([0, 1], size=10)
    return X, y


def test_corrected_nb_fitting_features_work(fixture_training_data):
    X, y = fixture_training_data

    clf = CorrectedNB()
    clf.fit(X, y)

    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    feat_count = np.zeros(shape=(n_classes, n_features))
    cls_count = np.zeros(shape=(n_classes,))
    for i, c in enumerate(np.unique(y)):
        feat_count[i] = X[y == c].sum(axis=0)
        cls_count[i] = len(y[y == c])

    assert (clf.feature_count_ == feat_count).all()
    assert (clf.class_count_ == cls_count).all()

    # test log prob is correct
    for feat_idx in np.random.randint(0, n_features, size=20):
        Ftot = clf.feature_count_[:, feat_idx].sum()
        pbases = clf.class_count_ / clf.class_count_.sum()
        for cls_idx, Fc in zip(
            range(len(clf.classes_)),
            clf.feature_count_[:, feat_idx],
        ):
            p_smooth = (Fc + 1) / (Ftot + 1 / (pbases[cls_idx]))
            p_norm_smooth = p_smooth / pbases[cls_idx]
            p_log_norm_smooth = np.log(p_norm_smooth)
            assert np.isclose(
                clf.feature_log_prob_[cls_idx, feat_idx],
                p_log_norm_smooth,
            )


def test_coverage_nb_fitting_features_work(fixture_training_data):
    X, y = fixture_training_data

    clf = CoverageNB()
    clf.fit(X, y)

    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    feat_count = np.zeros(shape=(n_classes, n_features))
    cls_count = np.zeros(shape=(n_classes,))
    for i, c in enumerate(np.unique(y)):
        feat_count[i] = X[y == c].sum(axis=0)
        cls_count[i] = len(y[y == c])

    assert (clf.feature_count_ == feat_count).all()
    assert (clf.class_count_ == cls_count).all()

    # test log prob is correct
    for feat_idx in np.random.randint(0, n_features, size=20):
        Ftot = clf.feature_count_[:, feat_idx].sum()
        pbases = clf.class_count_ / clf.class_count_.sum()
        for cls_idx, Fc in zip(
            range(len(clf.classes_)),
            clf.feature_count_[:, feat_idx],
        ):
            p_smooth = (Fc + 1) / (Ftot + 1 / (pbases[cls_idx]))
            p_norm_smooth = p_smooth / pbases[cls_idx]
            p_log_norm_smooth = np.log(p_norm_smooth)
            assert np.isclose(
                clf.feature_log_prob_[cls_idx, feat_idx],
                p_log_norm_smooth,
            )


def test_pipelinepilot_nb_fitting_features_work(fixture_training_data):
    X, y = fixture_training_data

    clf = PipelinePilotNB()
    clf.fit(X, y)

    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    feat_count = np.zeros(shape=(n_classes, n_features))
    cls_count = np.zeros(shape=(n_classes,))
    for i, c in enumerate(np.unique(y)):
        feat_count[i] = X[y == c].sum(axis=0)
        cls_count[i] = len(y[y == c])

    assert (clf.feature_count_ == feat_count).all()
    assert (clf.class_count_ == cls_count).all()

    # test log prob is correct
    for feat_idx in np.random.randint(0, n_features, size=20):
        Ftot = clf.feature_count_[:, feat_idx].sum()
        # PBases based on total feature counts rather than class counts.
        pbases = clf.feature_count_.sum(axis=1) / clf.feature_count_.sum()
        for cls_idx, Fc in zip(
            range(len(clf.classes_)),
            clf.feature_count_[:, feat_idx],
        ):
            p_smooth = (Fc + 1) / (Ftot + 1 / (pbases[cls_idx]))
            p_norm_smooth = p_smooth / pbases[cls_idx]
            p_log_norm_smooth = np.log(p_norm_smooth)
            assert np.isclose(
                clf.feature_log_prob_[cls_idx, feat_idx],
                p_log_norm_smooth,
            )
