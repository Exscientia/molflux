# Models

``molflux`` models can be stored into a standard format for packaging machine learning models that can be
used in a variety of our downstream tools - for example, real-time serving through a REST API or local batch inference.


## Storage Format

Ultimately, ``molflux`` models are a directory containing arbitrary files. Most importantly, they
include the serialised `molflux.modelzoo` model that you have trained, and metadata about
the `molflux.features` featurisation config applied at model training time to generate the model's input
features.

```shell
# Directory written by molflux.core.save_model(model, path="my_model", featurisation_metadata=...)
my_model/
├── model_config.json
├── model_artefacts/
|   └── model.pkl
├── featurisation_metadata.json
└── requirements.txt
```

### Additional Logged files

For environment recreation, we automatically log a `requirements.txt` file whenever a model is logged.
This file can then be used to reinstall dependencies using conda or virtualenv with pip.

## Models API

The productionising `.models` API lets you save and load ``molflux`` models ready for deployment. To save a model:

```python
from molflux.core.models import save_model

# model = <your-trained-model>
# featurisation_metadata = <your-featurisation-metadata>

save_model(model, path="my_model", featurisation_metadata=featurisation_metadata)
```

To load a model locally:

```python
from molflux.core.models import load_model

model = load_model(path="my_model")
```

## Scoring API

The productionising `.scoring` API implements a utility function for scoring your models on any given fold against an arbitrary
suite of metrics:

```python
from molflux.core.scoring import score_model

# model = <your-trained-model>
# fold = <the-fold-on-which-to-evaluate-the-model>
# metrics <the-metrics-to-use-to-evaluate-the-model>

scores = score_model(model, fold=fold, metrics=metrics)
```
