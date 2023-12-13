---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Uncertainty for models


Hopefully you have already read the [basic usage](basic_usage.md) and [ESOL tutorial](../tutorials/esol_training.md) and
are now ready to learn about how to use ``molflux.modelzoo`` models that provide uncertainty measures.

## Uncertainty

Uncertainty quantification is critical for ensuring trust in machine learning models and
enables techniques such as active learning by identifying which parts of a model contain the most uncertainty.

While every model in ``molflux.modelzoo`` acts as a basic estimator by defining
common functions such as `train(data, **kwargs)` and `predict(data, **kwargs)`, some regression
models implement additional functionalities:

  1) `predict_with_prediction_interval(data, confidence, **kwargs)` - returns prediction intervals along the predictions,
    ensuring a `confidence` that a prediction is within the corresponding interval
  2) `predict_std(data, **kwargs)` - returns standard deviation values along the predictions, as a measure of how
    uncertain the model is at each point
  3) `sample(data, n_samples, **kwargs)` - returns `n_samples` values for each input, drawn from the underlying
    distribution modelled for each input. For a given input, the average of the samples should be close to the prediction,
    while their spread indicates how uncertain the model is about this input.
  4) `calibrate_uncertainty(data, **kwargs)` - calibrates the uncertainty of this model to an external/validation dataset

You can check whether a model implements any of these methods by using the appropriate `supports_*` utility function:

```{code-cell} ipython3
from molflux.modelzoo import load_model, supports_prediction_interval

model = load_model(
  name="cat_boost_regressor",
  x_features=["x1", "x2"],
  y_features=["y"]
)

assert supports_prediction_interval(model)
```

Similarly, `supports_std`, `supports_sampling`, and `supports_uncertainty_calibration` are also available.


### Quick example - CatBoost Models

A typical example for a model with implemented uncertainty methods is the CatBoost Model. This model architecture can
return both a mean and standard deviation prediction.

In the example below, we will train and predict using a CatBoost, and then use some of the
functions defined above to get a measure of the model uncertainty.

```{code-cell} ipython3
import datasets
from molflux.modelzoo import load_model

model = load_model(
  name="cat_boost_regressor",
  x_features=["x1", "x2"],
  y_features=["y"]
)

train_dataset = datasets.Dataset.from_dict(
    {
        "x1": [0, 1, 2, 3, 4, 5],
        "x2": [0, -1, -2, -3, -4, -5],
        "y": [2, 4, 6, 8, 10, 12],
    }
)

model.train(train_dataset)

# Return just the predictions
print(model.predict(train_dataset))

# Return the predictions along a lower and upper bound to the 90% confidence interval
print(model.predict_with_prediction_interval(train_dataset, confidence=0.9))

# Return the predictions along the standard deviation
print(model.predict_with_std(train_dataset))
```


### Models calibrated with model-agnostic uncertainty

For models that do not have built in uncertainty available, we can make use of methods such as conformal prediction,
which provides a simple and effective way to create prediction intervals with guaranteed coverage probability
from any predictive model without making assumptions about the data-generating process or model.
Links to further resources on conformal prediction are
available [here](https://github.com/valeman/awesome-conformal-prediction)

There are two common patterns to generate and calibrate these prediction intervals:
1. In one go, during training - this is typically done via cross-validation and
under-the-hood the training data will be split up into `k` folds with a model
fitted `k` times.
2. In two steps - first, training an underlying model on training data, then
calibrating the uncertainty of it on a validation dataset

Both of these are possible with our [Mapie](https://github.com/scikit-learn-contrib/MAPIE) implementation.

```{note}
This functionality is still a work in progress.
```


#### 1) Mapie example - in one go
The main steps to get a model with calibrated uncertainty in this case are:
1. Instantiate a base modelzoo model object
2. Instantiate a mapie model
    - use the base estimator object as the `estimator` object
    - optionally, specify a value for `cv` except `prefit` (see the next example why)
3. Train the mapie model on any data
4. Use the model to generate calibrated prediction intervals on new data

```{code-cell} ipython3
import datasets
from molflux.modelzoo import load_model

# create a normal modelzoo model
original_model = load_model(
  name="random_forest_regressor",
  x_features=["x1", "x2"],
  y_features=["y"]
)

train_dataset = datasets.Dataset.from_dict(
    {
        "x1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "x2": [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        "y": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    }
)

# plug a mapie regressor on top
model = load_model("mapie_regressor",
    estimator=original_model,
    cv=5,
    x_features=original_model.x_features,
    y_features=original_model.y_features,
)
# train the model on new data
model.train(train_dataset)

model.predict_with_prediction_interval(train_dataset, confidence=0.9)
```

#### 2) Mapie example - in two steps

The main steps to get a model with calibrated uncertainty in this case are:
1. Instantiate a base modelzoo model object
2. Train the base model on some training data
3. Instantiate a mapie model
    - use the base, already trained object as the `estimator` object
    - set the `cv` argument as `prefit`
4. Calibrate the mapie model on some validation data
5. Use the model to generate calibrated prediction intervals on new data

```{code-cell} ipython3
import datasets
from molflux.modelzoo import load_model

# create a normal modelzoo model
original_model = load_model(
  name="random_forest_regressor",
  x_features=["x1", "x2"],
  y_features=["y"]
)

train_dataset = datasets.Dataset.from_dict(
    {
        "x1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "x2": [0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
        "y": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    }
)
validation_dataset = datasets.Dataset.from_dict(
    {
        "x1": [-10, -5, 0, 5, 10, 15],
        "x2": [10, 0, 0, 0, -10, -15],
        "y": [-18, -13, 2, 17, 22, 32],
    }
)

# train the original model on the training data
original_model.train(train_dataset)

# plug a mapie regressor on top, with the "prefit" option for "cv"
model = load_model("mapie_regressor",
    estimator=original_model,
    cv="prefit",
    x_features=original_model.x_features,
    y_features=original_model.y_features,
)

# calibrate the model on the validation data
model.calibrate_uncertainty(data=validation_dataset)

# predict on some new data (here, for simplicity, on the validation data)
model.predict_with_prediction_interval(data=validation_dataset, confidence=0.6)
```


### Conditional use of model-agnostic uncertainty

```{tip}
As mentioned above, there are a number of protocols that can be used to check if a
loaded model does support a specific uncertainty method.
This can then be used to allow conditional execution of code to wrap models that do not
support uncertainty with model agnostic uncertainty methods.
```

```{code-cell} ipython3
import datasets
from copy import copy
from molflux.modelzoo import load_from_dicts, load_model, supports_prediction_interval

list_of_configs = [
    {
        'name': 'random_forest_regressor',
        'config':
            {
                'x_features': ['x1', 'x2'],
                'y_features': ['y'],
                'n_estimators': 500,
                'max_depth': 10,
            },
    },
    {
        'name': 'cat_boost_regressor',
        'config':
            {
                'x_features': ['x1', 'x2'],
                'y_features': ['y'],
            },
    }
]

models = load_from_dicts(list_of_configs)

train_dataset = datasets.Dataset.from_dict(
    {
        "x1": [0, 1, 2, 3, 4, 5],
        "x2": [0, -1, -2, -3, -4, -5],
        "y": [2, 4, 6, 8, 10, 12],
    }
)

for original_model in models.values():

    if supports_prediction_interval(original_model):
        model = copy(original_model)
    else:
        # plug a mapie regressor on top
        model = load_model(
            "mapie_regressor",
            estimator=original_model,
            cv=5,
            x_features=original_model.x_features,
            y_features=original_model.y_features,
        )

    model.train(train_dataset)
    predictions, prediction_intervals = model.predict_with_prediction_interval(train_dataset, confidence=0.5)

    print(original_model.name, predictions, prediction_intervals)
```
