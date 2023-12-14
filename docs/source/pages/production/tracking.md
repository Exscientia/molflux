# Tracking

The ``molflux`` tracking API lets you log parameters, metrics, and other output files when running your machine
learning code.


## Tracking API

The productionising `.tracking` API lets you log a variety of artefacts associated with a given model training
experiment.

At the moment, the following generic utility functions are available:

``log_params()``

````{toggle}
Logs an arbitrary collection of parameters to disk as json.

```python

from molflux.core.tracking import log_params

# params = <arbitrary key-value pairs>

log_params(params, path="out/my_params.json")
```
````

``log_dataset()``

````{toggle}
Logs a dataset to disk.

```python
from molflux.core.tracking import log_dataset

# dataset = <arbitrary dataset>

log_dataset(dataset, path="out/my_dataset.parquet")
```
````

While these will allow you to log the corresponding objects according to standardised formats and conventions:


``log_pipeline_config()``

````{toggle}
Logs your pipeline config dictionary.

```python
from molflux.core.tracking import log_pipeline_config

# config = <your-pipeline-or-training-script-config>

log_pipeline_config(config, directory="out")
```
````


``log_featurised_dataset()``

````{toggle}
Logs a featurised dataset to disk.

```python
from molflux.core.tracking import log_featurised_dataset

# dataset = <arbitrary dataset>

log_featurised_dataset(dataset, directory="out")
```
````

``log_splitting_strategy()``

````{toggle}
Logs splitting strategy metadata to disk.

```python
from molflux.core.tracking import log_splitting_strategy

# splitting_strategy = <arbitrary splitting strategy>

log_splitting_strategy(splitting_strategy, directory="out")
```
````

``log_fold()``

````{toggle}
Logs a fold (DatasetDict) to disk.

```python
from molflux.core.tracking import log_fold

# fold = <arbitrary fold>

log_fold(fold, directory="out")
```
````

``log_model_params()``

````{toggle}
Logs your model metadata.

```python
from molflux.core.tracking import log_model_params

# model = <your-trained-model>

log_model_params(model, directory="out")
```
````

``log_scores()``

````{toggle}
Logs a nested dictionary of key-value metrics for each predictive task and for each fold.

```python
from molflux.core.scoring import score_model
from molflux.core.tracking import log_scores

# model = <your-model>
# fold = <your-dataset-dict>
# metrics = <your-metrics>

scores = score_model(model, fold=fold, metrics=metrics)
log_scores(scores, directory="out")
```
````
