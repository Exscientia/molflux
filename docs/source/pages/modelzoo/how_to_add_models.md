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

# Add your own model architecture

``molflux.modelzoo`` ships with a vast catalogue of available model-architectures, but you may want to experiment
at some point with your own model architecture, and make it more widely available to all users of
the package. In this guide, we will go through several ways of doing this, depending on the level of integration that
you would like to achieve.


## A model that quacks like a duck

To integrate with the rest of the ecosystem, your model architecture simply needs to behave in the same way as any
other `molflux.modelzoo` model architecture. To do this, you need to define a class that implements one or more of the
interfaces advertised by `molflux.modelzoo` in `molflux.modelzoo.protocols`.

For example, to define a new estimator, we simply need to fill in the `Estimator` protocol. For example:

```{code-cell} ipython3
import os
import pickle
from typing import Any, Dict, Optional

from molflux.modelzoo.typing import Features, DataFrameLike, PredictionResult


class MyEstimator:

    def __init__(
            self,
            tag: Optional[str] = "my_estimator",
            x_features: Optional[Features] = None,
            y_features: Optional[Features] = None,
            alpha: float = 0.5,
            **kwargs: Any
    ) -> None:
        self._tag = tag
        self._config = {
            "x_features": x_features,
            "y_features": y_features,
            "alpha": alpha,
        }
        self._model = None

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"tag": self._tag, "description": "My supercool model!"}

    @property
    def name(self) -> str:
        return "my_estimator"

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def x_features(self) -> Features:
        return self._config.get("x_features")

    @property
    def y_features(self) -> Features:
        return self._config.get("y_features")

    def train(self, train_data: DataFrameLike, **kwargs: Any) -> Any:
        self._model = "MODEL"

    def predict(self, data: DataFrameLike, **kwargs: Any) -> PredictionResult:
        predictions = ["tadaaa!" for _ in range(len(data))]
        return {f"{self._tag}::{y_feature}": predictions for y_feature in self.y_features}

    def as_dir(self, directory: str) -> None:
        pickle_fn = os.path.join(directory, "model.pkl")
        with open(pickle_fn, "wb") as f:
            pickle.dump(self._model, f)

    def from_dir(self, directory: str) -> None:
        pickle_fn = os.path.join(directory, "model.pkl")
        with open(pickle_fn, "rb") as f:
            model = pickle.load(f)

        self._model = model

    def __str__(self) -> str:
        return "This is my supercool model!"
```

```{note}
Note that we made use of some convenience methods from molflux.modelzoo to annotate our model, but you could / should technically
fully define your class without even importing `molflux.modelzoo` at all!
```

And you can check that your new model architecture does indeed implement the correct protocol as follows:
```{code-cell} ipython3
from molflux.modelzoo.protocols import Estimator

ok = isinstance(MyEstimator, Estimator)
print(ok)
```

If everything went well, you can now use your model architecture anywhere a `molflux.modelzoo` model would be expected!

This first step is useful if you are still prototyping and iterating on your model. You can now easily swap it in
anywhere in the ``molflux`` ecosystem, see what works, see what doesn't, and iterate on your model architecture.

## Adding your model to your local catalogue

You may have noticed that while you can now feed your model to any function expecting `molflux.modelzoo` models, it does not
show up yet in the `molflux.modelzoo` catalogue of available model architectures.

To do this, you can decorate your class as follows:

```{code-block} python
from molflux.modelzoo import register_model


@register_model(kind="custom", name="my_estimator")
class MyEstimator:
    ...
```

Et voilà ! The model now appears in the catalogue (as `my_estimator`, under a `custom` category), and you can load it
like any other native model architecture:

```python
from molflux.modelzoo import list_models, load_model

catalogue = list_models()
print(catalogue)
# {..., 'custom': ['my_estimator'], ...}

model = load_model("my_estimator")
print(model)
```

This step is useful once you have tested the low-level behaviour of your model, and you would like to test the
higher level config-driven integration with the rest of the ``molflux`` ecosystem.

## Sharing your model

If you have settled on a model architecture and would like to share it with the wider `molflux` community, you now
have two options:

1. Create a python package to distribute your model architecture class
2. Contribute your class directly to the `molflux` codebase (opening a PR)

Both approaches have their pros and cons. The former will give you much more flexibility over your
model architecture and its development as you will be completely decoupled from any `molflux.modelzoo` changes (other than
those to the public protocol API). It does mean though that you will now have one more piece of software to maintain,
and you may end up re-implementing many of the utility features that already exist in `molflux.modelzoo`.

### Publishing your model as a plugin

Note that if shipping your model architecture as part of your own package, clients will only have access to your model
if they have installed your package in their python environment. In addition, your model will get registered in the
``molflux.modelzoo`` catalogue only if the module defining your model architecture class has been imported explicitly.

To avoid the latter point, we strongly recommended that you register your model as a plugin (to learn more about plugins see
[here](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)). Using this method, your model
will get registered automatically as long as your users have installed your package in their environment.

To begin with, remove the ``register_model`` decorator from your model architecture class definition
(it's no longer needed):

```{code-block} python
from molflux.modelzoo import register_model


@register_model(kind="custom", name="my_estimator")
class MyEstimator:
    ...
```


Now, go to the ``pyproject.toml`` file of your repo and under ``[project.entry-points.'molflux.modelzoo.plugins.pluging_name']``,
add a plugin to your model class as follows

```{code-block} ini
[project.entry-points.'molflux.modelzoo.plugins.plugin_name']
name_of_model = 'path.to.module.file:YourModelName'
```

```{note}
You can also do this in the ``setup.cfg`` file of your repo and under ``[options.entry_points]``. Add a plugin to your
model class as follows

```{code-block} ini
[options.entry_points]
molflux.modelzoo.plugins.kind_of_model =
   name_of_model = path.to.module.file:YourModelName
```

This entry point allows ``molflux.modelzoo`` to hook into your package and automatically register your model in the catalogue.
In the plugin definition, you need to specify the `kind` and `name` of your model, pointing at the path to your
model architecture class definition. To make discovery in the catalogue easier, set your model architecture's `kind` to
something that can easily be associated with your package.

And that's it! Anyone will now be able to load your model from the `molflux.modelzoo` catalogue!
