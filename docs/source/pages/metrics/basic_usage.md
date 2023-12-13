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

In this section, we will illustrate how to use ``molflux.metrics``. These examples will provide you with a starting
point.


## Browsing

First, we'll review which metrics are available for use. These are conveniently categorised (for example,
into ``regression``, ``classification``, etc.). To view what's available you can do

```{code-cell} ipython3

from molflux.metrics import list_metrics

catalogue = list_metrics()

print(catalogue)
```

This returns a dictionary of available metrics (organised by categories and ``name``). There are a few to choose from.
To see how you can add your own metrics, see [How to add your own metrics](how_to_add_metrics.md).

When computing metrics, it is often useful to compute all of the possible regression or classification metrics to get a better
idea of the model performance. To this end, ``molflux.metrics`` provides you with the option to load an entire metric suite.
To view what suites are available, you can do

```{code-cell} ipython3

from molflux.metrics import list_suites

catalogue = list_suites()

print(catalogue)
```


## Loading metrics

Loading a ``molflux.metrics`` metric is very easy, simply do

```{code-cell} ipython3

from molflux.metrics import load_metric

metric = load_metric(name="r2")

print(metric)
```

By printing the loaded metric, you get more information about it. Each metric has a ``name``, and a ``tag``
(to uniquely identify it in case you would like to generate multiple copies of the same metric but with different
configurations). You can also see the optional compute metric arguments (and their default values) in the signature.
There is also a short description of the metric.

To load a metric suite, you can do
```{code-cell} ipython3

from molflux.metrics import load_suite

suite = load_suite("regression")

print(suite)
```

You can also load a metric from a config dictionary. A metrics config dictionary is a dictionary specifying the metric to be
loaded. A config dictionary must have the following format
```{code-block} python
metrics_dict = {
  'name': '<name of the metric>',
  'config': '<kwargs for instantiating metric>'
  'presets': '<kwarg presets for computing metric>'
}
```

The ``name`` key specifies the ``name``  of the metric to load from the catalogue. The ``config`` key
specifies the arguments that are needed for instantiating the metric
and the ``presets`` key specifies some preset kwargs to apply on computing the metric. If neither
is specified, the metric will use default values.

To load a metric from a config

```{code-cell} ipython3

from molflux.metrics import load_from_dict

config = {
    'name': 'r2',
    'presets': {
        'sample_weight': [0.2, 0.4, 0.4],
    },
}

metric = load_from_dict(config)

print(metric.state)
```

You can also load multiple metrics all at once using a list of config dictionaries. This is done as follows


```{code-cell} ipython3

from molflux.metrics import load_from_dicts

list_of_configs = [
    {
        'name': 'r2',
        'presets':
        {
            'sample_weight': [0.2, 0.4, 0.4],
        },
    },
    {
        'name': 'mean_squared_error',
    }
]

metrics = load_from_dicts(list_of_configs)

print(metrics)
```


Finally, you can load metrics from a yaml file. You can use a single yaml file which includes configs for all the ``molflux`` tools,
and ``molflux.metrics`` will know how to extract the relevant part it needs. To do so, you need to define a yaml file with the
following example format

```{code-block} yaml

---
version: v1
kind: metrics
specs:
    - name: r2
      presets:
        sample_weight: [0.2, 0.4, 0.4]
    - name: mean_squared_error
...

```
It consists of a version (this is the version of the config format, for now just ``v1``), ``kind`` of config (in this case
``metrics``), and ``specs``. ``specs`` is where the keyword arguments used when computing the metric are defined. The
yaml file can include configs for other ``molflux`` modules as well. To load this yaml file, you can simply do


```{code-block} ipython3

from molflux.metrics import load_from_yaml

metrics = load_from_yaml(path_to_yaml_file)

print(metrics)
```


## Computing metrics

After loading a metric (or group of metrics), you can apply them to predictions to compute the metric values. You must pass
the references (ground truths) and the predictions (from your model).

```{code-cell} ipython3

from molflux.metrics import load_metric

metric = load_metric("r2")

ground_truth = [0, 0.3, 0.5, 0.8, 1]
preds = [0.1, 0.35, 0.45, 0.68, 1.2]

results = metric.compute(predictions=preds, references=ground_truth)

print(results)
```

This will return a dictionary with the metric ``tag`` as the key and the computed metric as the value. For a group
of metrics (or a metric suite), you can follow the same procedure

```{code-cell} ipython3

from molflux.metrics import load_from_dicts

list_of_configs = [
    {
        'name': 'r2',
        'presets':
        {
            'sample_weight': [0.2, 0.2, 0.3, 0.1, 0.2],
        },
    },
    {
        'name': 'mean_squared_error',
    }
]

metrics = load_from_dicts(list_of_configs)

ground_truth = [0, 0.3, 0.5, 0.8, 1]
preds = [0.1, 0.35, 0.45, 0.68, 1.2]

results = metrics.compute(predictions=preds, references=ground_truth)

print(results)

```

This will return a dictionary with all the computed metrics (where the ``tags`` as the keys and the metrics as the values).
