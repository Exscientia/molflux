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

# Standardisation


A driving principle of the ``molflux`` package is standardisation across all of its five submodules:
`datasets`, `features`, `splits`, `modelzoo`, `metrics`. In this section, we demonstrate and describe the unified
API of the package. Learning to use the following functionality gives the user immediate knowledge of how to use all parts
of the package. Each of the following methods can be imported from the relevant submodule (for example
`from molflux.datasets import list_datasets`).

## Browsing

To start, we first introduce the basic browsing functionality of the submodules. Each submodule has a ``list_*`` function
that returns a dictionary of available objects (datasets, representations, models, etc...). These are

1) ``list_datasets``
2) ``list_representations``
3) ``list_splits``
4) ``list_models``
5) ``list_metrics``

The dictionaries returned are grouped by the optional dependency required for the objects (key) and the list of available
objects (value). For more information, see the respective browsing sections in the \
documentation:  [datasets](../datasets/basic_usage.md#browsing),
                [features](../features/basic_usage.md#browsing),
                [splits](../splits/basic_usage.md#browsing),
                [modelzoo](../modelzoo/basic_usage.md#browsing),
                [metrics](../metrics/basic_usage.md#browsing).

## Loading

### ``load_*``

The ``load_*`` is the most straightforward loading method of the API. It can be used to load the objects of any of the
five main submodules

1) ``load_datasets``
2) ``load_representation``
3) ``load_splitting_strategy``
4) ``load_model``
5) ``load_metric``

The pattern is ``load_*(name: str, **kwargs)`` where ``name`` is a string name of the object and ``kwargs`` are optional
object specific kwargs.

### ``load_from_dict``

Although the ``load_*`` function can load the required objects, specifying kwargs directly can become tedious. In general,
we recommend using a config driven approach to load objects. This is done via the ``load_from_dict`` method which expects
a config dictionary in the following format

```{code-block} python
{
  "name": <string name of object>,
  "config": <dictionary of kwargs used at instantiation>,
  "presets": <dictionary of kwargs used at the relevant method call>,
}
```

### ``load_from_dicts``

It is not uncommon to want to load multiple objects of the same flavour at the same type (for example loading multiple datasets
or multiple representations). For this, we have the ``load_from_dicts`` (plural) which takes in a list of config dictionaries
as specified in ``load_from_dict``.


### ``load_from_yaml``

A complete machine learning pipeline consists of multiple stages each of which requires a config. To streamline the process
of specifying configs for pipelines, we provide the option to use a unified ``yaml`` file to store the configs of
all the stages and the ``load_from_yaml`` for each submodule which knows how to pick out the relevant parts of the config
for its purposes.

The general form of the yaml config looks like

```{code-block} yaml

---
version: v1
kind: datasets
specs:
  - name:
    config:
---
version: v1
kind: representations
specs:
  - name:
    config:
---
version: v1
kind: splits
specs:
  - name:
    config:
---
version: v1
kind: models
specs:
  - name:
    config:
---
version: v1
kind: metrics
specs:
  - name:
    config:
```

where each submodule can take a list of configs for each object to load.

```{seealso}
For an explicit example of using a single yaml file, see the [ESOL regression example using yaml](../tutorials/esol_reg.md#esol-training-using-a-yaml-file).
```

For more information, see the respective loading sections in the \
documentation:  [datasets](../datasets/basic_usage.md#loading-datasets),
                [features](../features/basic_usage.md#loading-representations),
                [splits](../splits/basic_usage.md#loading-splitting-strategies),
                [modelzoo](../modelzoo/basic_usage.md#loading-a-model-architecture),
                [metrics](../metrics/basic_usage.md#loading-metrics).
