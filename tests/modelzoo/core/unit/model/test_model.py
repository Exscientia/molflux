import pandas as pd
import pytest

import datasets
from molflux.modelzoo.load import load_model
from molflux.modelzoo.protocols import Estimator


def test_load():
    model = load_model("mock_model")
    assert model is not None


def test_implements_protocol():
    model = load_model("mock_model")
    assert isinstance(model, Estimator)


train_df = pd.DataFrame({"x1": ["Hello"], "y1": ["Buongiorno"]})


@pytest.mark.parametrize(
    "train_data",
    [
        train_df,
        datasets.Dataset.from_pandas(train_df),
    ],
)
def test_train(train_data):
    model = load_model("mock_model", x_features=["x1"], y_features=["y1"])
    model.train(train_data=train_data)
    assert True
