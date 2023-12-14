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

# Basic usage


In this section, we will illustrate how to use ``molflux.modelzoo``. These examples will provide you with a starting
point.


## Browsing

First, we review which what model architectures are available for use. To view what's available you can do

```{code-cell} ipython3

from molflux.modelzoo import list_models

catalogue = list_models()

print(catalogue)
```

This returns our catalogue of available model architectures (organised by the dependencies they rely on). There are a few to choose from.

```{seealso}
[How to add your own model](how_to_add_models.md) if you would like to add your own model to the catalogue
```
 For instance, `molflux.modelzoo.list_models()` returns as one item in the dictionary:
`'xgboost': ['xg_boost_classifier', 'xg_boost_regressor']`. In order to be able to use the two models `xg_boost_classifier`
and `xg_boost_regressor`, you would do: ``pip install molflux[xgboost]``.

## Loading a model architecture

Loading a model architecture of your choice is simple. For example, to load a `random_forest_regressor` from the
catalogue:

```{code-cell} ipython3

from molflux.modelzoo import load_model

model = load_model(name="random_forest_regressor")

print(model)
```

By printing the loaded model architecture, you get more information about it. Each model has a ``name``, a ``tag``
(to uniquely identify it in case you would like to generate multiple copies of the same model but with different
configurations), and a set of architecture-specific configuration parameters. You should also be able to view a
short description of the model, and get some extra information about the model's method signatures.

To load a model with non-default configuration parameters, you can simply supply them at load time:

```{code-cell} ipython3

from molflux.modelzoo import load_model

model = load_model(
  name="random_forest_regressor",
  tag="my_rf",
  x_features=["x1", "x2"],
  y_features=["y"],
  n_estimators=50
)

# double check your model's architecture configuration
print(model.config)
```

With time, you may want to load model architectures using a config-driven approach. To do this, `molflux.modelzoo` supports
loading model architectures from dictionaries specifying the model architecture to be loaded and its configuration parameters:

```python

from molflux.modelzoo import load_from_dict

config = {
            'name': 'random_forest_regressor',
            'config':
                {
                    'tag': "my_rf",
                    'x_features': ['x1', 'x2'],
                    'y_features': ['y'],
                    'n_estimators': 50,
                },
        }

model = load_from_dict(config)
```

The ``name`` key specifies the ``name``  of the model architecture to load from the catalogue. The ``config`` key
should hold the dictionary of configuration arguments to initialise the model with (if not specified, the model will use default values).

You can also load multiple models all at once using a list of config dictionaries. This is done as follows

```{code-cell} ipython3

from molflux.modelzoo import load_from_dicts

list_of_configs = [
    {
            'name': 'random_forest_regressor',
            'config':
                {
                    'tag': "my_rf_1",
                    'x_features': ['x1', 'x2'],
                    'y_features': ['y'],
                    'n_estimators': 500,
                    'max_depth': 10,
                },
        },
    {
            'name': 'random_forest_classifier',
            'config':
                {
                    'tag': "my_rf_2",
                    'x_features': ['x1', 'x2'],
                    'y_features': ['y'],
                    'n_estimators': 500,
                    'max_depth': 10,
                },
        }
]

models = load_from_dicts(list_of_configs)

print(models)
```


Finally, you can load models from a yaml file. You can use a single yaml file which includes configs for all the ``molflux`` tools
and ``molflux.modelzoo`` will know how to extract the relevant part it needs. To do so, you need to define a yaml file with the
following example format

```{code-block} yaml

---
version: v1
kind: models
specs:
    - name: random_forest_regressor
      config:
        tag: my_rf_1
        x_features:
            - x1
            - x2
        y_features:
            - y1
        n_estimators: 500
    - name: random_forest_classifier
      config:
        tag: my_rf_2
        x_features:
            - x1
            - x2
        y_features:
            - y1
        n_estimators: 300
...

```
It consists of a version (this is the version of the config format, for now just ``v1``), ``kind`` of config (in this case
``models``), and ``specs``. ``specs`` is where the configs are defined. The yaml file can include
configs for other ``molflux`` modules as well. To load the model from the yaml file, you can simply do


```{code-block} ipython3

from molflux.modelzoo import load_from_yaml

models = load_from_yaml(path_to_yaml_file)

print(models)
```


## Training/Inferencing a model

All models in ``molflux.modelzoo`` have ``train`` and ``predict`` methods. These are the main two methods you need to
interact with.

### Training

After loading a model architecture, you can train it on a dataset using the model's `train()` method, to which you
should feed your training dataset and optional training arguments (if any are specified by the model architecture
of your choice).

```{note}
Our model's interfaces accept dataframe-like objects that implement the
[Dataframe Interchange Protocol](https://data-apis.org/dataframe-protocol/latest/purpose_and_scope.html) as input data:
these include pandas dataframes, pyarrow tables, vaex dataframes, cudf dataframes, and many other popular dataframe
libraries... We also support HuggingFace [datasets](https://huggingface.co/docs/datasets/index) as inputs for seamless
integration with our datasets users. If you are used to working with other in-memory data representations,
you will need to convert them before feeding them to our models. Please [contact us](https://github.com/Exscientia/molflux/issues) if you need support with,
your workflows.
```

For example, we can train our `random_forest_regressor` as follows:

```{code-block} ipython3
import datasets
from molflux.modelzoo import load_model

model = load_model(
  name="random_forest_regressor",
  x_features=["x1", "x2"],
  y_features=["y"],
  n_estimators=50
)

train_data = datasets.Dataset.from_dict(
    {
        "x1": [0, 1, 2, 3, 4, 5],
        "x2": [0, -1, -2, -3, -4, -5],
        "y": [2, 4, 6, 8, 10, 12],
    }
)

model.train(train_data)
```

And the model is trained!

A pandas dataframe would have also worked in this case - although we recommend switching to dataframe libraries backed by
apache arrow (like `pyarrow`, or `datasets` shown above), as not all pandas column dtypes can be cast to arrow:

```{code-block} ipython3
import pandas as pd
from molflux.modelzoo import load_model

model = load_model(
  name="random_forest_regressor",
  x_features=["x1", "x2"],
  y_features=["y"],
  n_estimators=50
)

train_data = pd.DataFrame(
    {
        "x1": [0, 1, 2, 3, 4, 5],
        "x2": [0, -1, -2, -3, -4, -5],
        "y": [2, 4, 6, 8, 10, 12],
    }
)

model.train(train_data)
```


```{tip}
To disable progress bars you can call `datasets.disable_progress_bar()` anywhere in your script.
```

### Inferencing

Once a model is trained, you can use it for inference using the model's `predict()` method, to which you should
feed the dataset you would like to get predictions for:

```{code-block} ipython3
import datasets

test_data = datasets.Dataset.from_dict(
    {
        "x1": [10, 12],
        "x2": [-2.5, -1]
    }
)

predictions = model.predict(test_data)
print(predictions)
```

This returns a dictionary of your model's predictions! Models can also support different inference methods. For example,
some classification models support the ``predict_proba`` method which returns the probabilities of the classes

```python
probabilities = model.predict_proba(test_data)
```

## Saving/Loading a model

Once you have trained your model, you can save it and load it for later use.

### Saving

To save a model, all you have to do is

```{code-block} python
from molflux.modelzoo import save_to_store

save_to_store("path_to_my_model/", model)
```

The ``save_to_store`` function takes the path and the model to save. It can save to local disk or to an s3 location.

```{note}
For models intended for production level usage, we recommend that they are saved as described in the [productionising](../production/models.md)
section. Along with the model, this also saves the featurisation metadata and a snapshot of the environment the model was
built in.
```

### Loading

To load, you simply need to do

```{code-block} python
from molflux.modelzoo import load_from_store

model = load_from_store("path_to_my_model/")
```

This can load from local disk and s3.
