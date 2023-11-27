# Tracking

The ``molflux`` tracking API lets you log parameters, metrics, and other output files when running your machine
learning code.

```{contents} On this page
---
local: true
backlinks: none
---
```

## Tracking API

The productionising `.tracking` API lets you log a variety of artefacts associated with a given model training
experiment.

At the moment, the following generic utility functions are available:

:::{confval} log_params()
Logs an arbitrary collection of parameters to disk as json.

```python
from molflux.core.tracking import log_params

# params = <arbitrary key-value pairs>

log_params(params, path="out/my_params.json")
```
:::

:::{confval} log_dataset()
Logs an dataset to disk.

```python
from molflux.core.tracking import log_dataset

# dataset = <arbitrary dataset>

log_dataset(dataset, path="out/my_dataset.parquet")
```
:::

While these will take care for you to log the corresponding objects according to standardised formats and conventions:

::: {confval} log_pipeline_config()
Logs your pipeline config dictionary.

```python
from molflux.core.tracking import log_pipeline_config

# config = <your-pipeline-or-training-script-config>

log_pipeline_config(config, directory="out")
```
:::

:::{confval} log_featurised_dataset()
Logs a featurised dataset to disk.

```python
from molflux.core.tracking import log_featurised_dataset

# dataset = <arbitrary dataset>

log_featurised_dataset(dataset, directory="out")
```
:::

:::{confval} log_splitting_strategy()
Logs splitting strategy metadata to disk.

```python
from molflux.core.tracking import log_splitting_strategy

# splitting_strategy = <arbitrary splitting strategy>

log_splitting_strategy(splitting_strategy, directory="out")
```
:::

:::{confval} log_fold()
Logs a fold (DatasetDict) to disk.

```python
from molflux.core.tracking import log_fold

# fold = <arbitrary fold>

log_fold(fold, directory="out")
```
:::

::: {confval} log_model_params()
Logs your model metadata.

```python
from molflux.core.tracking import log_model_params

# model = <your-trained-model>

log_model_params(model, directory="out")
```
:::

:::{confval} log_scores()
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
:::
