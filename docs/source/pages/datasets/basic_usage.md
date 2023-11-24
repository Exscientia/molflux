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

In this section, we will quickly illustrate how to use ``molflux.datasets``. These examples will provide you with a starting
point. Much of the low level functionality is already documented in the ``huggingface`` docs (LINK). Here, we will only go
through the added functionality.


## Exploration

First, let's have a look at what datasets are available for use. To view what's available, you can run the `list_datasets` function:

```{code-cell} ipython3

from molflux.datasets import list_datasets

catalogue = list_datasets()

print(catalogue)
```

This returns a list of available datasets by their name.

## Loading datasets

Loading a dataset is very simple. You just need to run `load_dataset` with a given dataset name:

```{code-cell} ipython3

from molflux.datasets import load_dataset

dataset = load_dataset('esol')
print(dataset)
```

By printing the loaded dataset, you can see minimal information about it like the column names and number of datapoints.

You can also load a dataset from a config. A dataset config is a dictionary specifying the dataset to be
loaded. A config dictionary must have the following format
```{code-block} python
dataset_config_dict = {
    'name': '<name of the dataset>',
    'config': '<kwargs for instantiating dataset>'
}
```

The ``name`` key specifies the ``name``  of the dataset to load from the catalogue. The ``config`` key
specifies the arguments that are needed for instantiating the dataset.

To load a dataset from a config

```{code-cell} ipython3

from molflux.datasets import load_from_dict

config = {
    'name': 'esol',
}

dataset = load_from_dict(config)
print(dataset)
```

For convenience, you can also load a group of dataset all at once by specifying a list of configs.

```{code-cell} ipython3

from molflux.datasets import load_from_dicts

config = [
        {
            'name': 'esol',
        },
        {
            'name': 'esol',
        }
]

datasets = load_from_dicts(config)
print(datasets)
```

Finally, you can load datasets from a yaml file. You can use a single yaml file which includes configs for all the ``molflux``
submodules and the ``molflux.datasets.load_from_yaml`` will know how to extract the relevant part it needs. To do so, you need to define a yaml file with the
following example format

```{code-block} yaml

---
version: v1
kind: datasets
specs:
    - name: esol
...

```
It consists of a version (this is the version of the config format, for now just ``v1``), ``kind`` of config (in this case
``datasets``), and ``config``. ``config`` is where the dataset initialisation keyword arguments are defined. The yaml file can include
configs for other ``molflux`` packages as well. To load this yaml file, you can simply do


```{code-block} ipython3

from molflux.datasets import load_from_yaml

datasets = load_from_yaml(path_to_yaml_file)

print(datasets)
```

## Working with datasets

``molflux.datasets`` was designed to supplement the HuggingFace [datasets](https://huggingface.co/docs/datasets/index) library,
giving you access to our internal catalogue of datasets, and to a number of convenience utility functions. The datasets
returned by e.g. `molflux.datasets.load_dataset()` are actually native huggingface datasets, with all of the associated
functionalities.

You can find complete documentation on how to work with hugginface datasets [online](https://huggingface.co/docs/datasets/index), or
even check out their official [training course](https://huggingface.co/course/chapter5/1?fw=pt)! The rest of this
tutorial will simply show you a couple of examples of some of the most basic functionalities available.

You can inspect individual datapoints and get the column names:

```{code-cell} ipython3

from molflux.datasets import load_dataset

dataset = load_dataset('esol')

print(dataset[123])
print(dataset.column_names)
```

Adding columns is easily done by:

```{code-cell} ipython3

from molflux.datasets import load_dataset

dataset = load_dataset('esol')

dataset = dataset.add_column("my_new_column", list(range(len(dataset))))

print(dataset)
```

You can also transform the dataset into a pandas DataFrame:

```{code-cell} ipython3

from molflux.datasets import load_dataset

dataset = load_dataset('esol')

print(dataset.to_pandas())
```

You can also save and load the datasets to disk or to the cloud (s3)

```{code-block} python
from molflux.datasets import load_dataset, load_dataset_from_store, save_dataset_to_store

dataset = load_dataset('esol')

save_dataset_to_store(dataset, "my/data/dataset.parquet")

dataset = load_dataset_from_store("my/data/dataset.parquet")
```

```{seealso}

For more information on how to save and load datasets to disk, checkout these two how to guides:
* [How to save datasets](saving.md)
* [How to load datasets](loading.md)
```
