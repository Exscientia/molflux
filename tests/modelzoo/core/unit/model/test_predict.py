import pandas as pd
import pytest

import datasets
from molflux.modelzoo.load import load_model

df = pd.DataFrame({"x1": ["Hello"], "y1": ["Buongiorno"]})


@pytest.mark.parametrize(
    "data",
    [
        df,
        datasets.Dataset.from_pandas(df),
    ],
)
def test_predict_from_dataframe(data):
    """That can predict using any dataframe-like object that implements the DataFrame Interchange Protocol."""
    model = load_model("mock_model", x_features=["x1"], y_features=["y1"])
    model.train(data)
    prediction = model.predict(data=data)
    assert prediction == {model.tag: "Spooky Result!"}


def test_invalid_kwarg_raises_value_error():
    """That attempting prediction with invalid keyword arguments raises."""
    model = load_model("mock_model", x_features=["x1"])
    data = datasets.Dataset.from_dict({"x1": ["Hello"], "y1": ["Buongiorno"]})
    model.train(data)
    with pytest.raises(ValueError, match=r"Unknown predict parameter\(s\)"):
        model.predict(data, invalid_kwarg=True)
