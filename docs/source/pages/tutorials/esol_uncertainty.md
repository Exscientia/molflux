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

# ESOL uncertainty

In this tutorial we provide a simple example of training a random forest model on the [ESOL dataset](https://pubs.acs.org/doi/10.1021/ci034243x),
and adding an uncertainty estimator using [Mapie](https://github.com/scikit-learn-contrib/MAPIE) on top. We require the ``rdkit``
and ``mapie`` packages, so make sure to ``pip install 'molflux[rdkit,mapie]'`` to follow along!


## Loading the ESOL dataset

First, let's load the ESOL dataset

```{code-cell} ipython3

from molflux.datasets import load_dataset

dataset = load_dataset("esol")

print(dataset)

dataset[0]
```

The loaded dataset is an instance of a HuggingFace ``Dataset`` (for more info, see the [docs](https://huggingface.co/docs/datasets/index)).
You can see that there are two columns: ``smiles`` and ``log_solubility``.


## Featurising

Now, we will featurise the dataset. For this, we will use the Morgan and MACCS fingerprints from ``rdkit`` and the
``featurise_dataset`` function from ``molflux.datasets``.

```{code-cell} ipython3

from molflux.datasets import featurise_dataset
from molflux.features import load_from_dicts as load_representations_from_dicts

featuriser = load_representations_from_dicts(
    [
        {"name": "morgan"},
        {"name": "maccs_rdkit"},
    ]
)

featurised_dataset = featurise_dataset(dataset, column="smiles", representations=featuriser)

print(featurised_dataset)
```

You can see that we now have two extra columns for each fingerprint we used.

## Splitting

Next, we need to split the dataset. For this, we use the simple ``shuffle_split`` (random split) with 70% training,
10% validation, and 20% test. To split the dataset, we use the ``split_dataset`` function from ``molflux.datasets``.

```{code-cell} ipython3

from molflux.datasets import split_dataset
from molflux.splits import load_from_dict as load_split_from_dict

shuffle_strategy = load_split_from_dict(
    {
        "name": "shuffle_split",
        "presets": {
            "train_fraction": 0.7,
            "validation_fraction": 0.1,
            "test_fraction": 0.2,
        }
    }
)

split_featurised_dataset = next(split_dataset(featurised_dataset, shuffle_strategy))

print(split_featurised_dataset)
```


## Training the model

We can now turn to training the model! We choose the ``random_forest_regressor`` (which we access from the ``sklearn`` package).
To do so, we need to define the model config and the ``x_features`` and the ``y_features``.

Once trained, we will get some predictions and compute some metrics!

```{code-cell} ipython3
import json

from molflux.modelzoo import load_from_dict as load_model_from_dict
from molflux.metrics import load_suite

import matplotlib.pyplot as plt

model = load_model_from_dict(
    {
        "name": "random_forest_regressor",
        "config": {
            "x_features": ['smiles::morgan', 'smiles::maccs_rdkit'],
            "y_features": ['log_solubility'],
        }
    }
)

model.train(split_featurised_dataset["train"])

preds = model.predict(split_featurised_dataset["test"])

regression_suite = load_suite("regression")

scores = regression_suite.compute(
    references=split_featurised_dataset["test"]["log_solubility"],
    predictions=preds["random_forest_regressor::log_solubility"],
)

print(json.dumps({k: round(v, 2) for k, v in scores.items()}, indent=4))

plt.scatter(
    split_featurised_dataset["test"]["log_solubility"],
    preds["random_forest_regressor::log_solubility"],
)
plt.plot([-9, 2], [-9, 2], c='r')
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
```

## Adding a Mapie uncertainty estimator on top

Once the random forest model is trained, we can build a Mapie estimator on top. For more information, check out the [uncertainty
for models](../modelzoo/uncertainty.md) section.

To build the Mapie model, you can first train a random forest model and then calibrate a Mapie model on the validation
set as follows

```{code-cell} ipython3

mapie_model = load_model_from_dict(
    {
        "name": "mapie_regressor",
        "config": {
            "x_features": model.x_features,
            "y_features": model.y_features,
            "estimator": model,
            "cv": "prefit",
        }
    }
)

mapie_model.calibrate_uncertainty(split_featurised_dataset["validation"])

preds, intervals = mapie_model.predict_with_prediction_interval(
    split_featurised_dataset["test"],
    confidence=0.9,
)

xs = split_featurised_dataset["test"]["log_solubility"]
ys = preds["mapie_regressor::log_solubility"]
y_intervals = intervals["mapie_regressor::log_solubility::prediction_interval"]

yerrs = [
    [abs(y - y_in[0])  for y, y_in in zip(ys, y_intervals)],
    [abs(y - y_in[1])  for y, y_in in zip(ys, y_intervals)],
]

plt.errorbar(
    x=xs,
    y=ys,
    yerr=yerrs,
    fmt='o',
)
plt.plot([-9, 2], [-9, 2], c='r')
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
```

And finally you get some error bars!
