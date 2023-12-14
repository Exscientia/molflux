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
point. Much of the low level functionality is already documented in the HuggingFace ``dataset`` [docs](https://huggingface.co/docs/datasets/index).
Here, we will go through the basics and the added functionality from ``molflux``.


## Browsing

First, we use the ``list_datasets`` function to browse what datasets are available.

```{code-cell} ipython3

from molflux.datasets import list_datasets

catalogue = list_datasets()

print(catalogue)
```

This returns a list of available datasets by their name.

```{tip}
On top of these drug discovery datasets, you can also access all datasets from the HuggingFace [registry](https://huggingface.co/datasets) (for example,
the MNIST dataset). Follow along with the rest of this page with your favourite dataset from there!
```

## Loading datasets

### Loading using ``load_dataset``

Loading a dataset is simple. You just need to run `load_dataset` with a given dataset name:

```{code-cell} ipython3

from molflux.datasets import load_dataset

dataset = load_dataset('esol')
print(dataset)
```

By printing the loaded dataset, you can see minimal information about it like the column names and number of datapoints.

```{tip}
You can also see more information about the dataset from its ``dataset.info``.
```

### Loading using ``load_from_dict``

Datasets can also be loaded by specifying a config dictionary. A config dictionary must have the following format
```{code-block} python
dataset_config_dict = {
    'name': '<name of the dataset>',
    'config': '<kwargs for instantiating dataset>'
}
```

The ``name`` key specifies the ``name``  of the dataset to load from the catalogue. The ``config`` key
specifies the arguments that are needed for instantiating the dataset. The dataset can then be loaded by doing

```{code-cell} ipython3

from molflux.datasets import load_from_dict

config = {
    'name': 'esol',
}

dataset = load_from_dict(config)
dataset
```

### Loading using ``load_from_dicts``

For convenience, you can also load a group of datasets all at once by specifying a list of configs.

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

### Loading using ``load_from_yaml``

Finally, you can load datasets from a yaml file. You can use a single yaml file which includes configs for all the ``molflux``
submodules, and the ``molflux.datasets.load_from_yaml`` will know how to extract the relevant part it needs for the dataset.
To do so, you need to define a yaml file with the following example format

```{code-block} yaml

---
version: v1
kind: datasets
specs:
    - name: esol
...

```
It consists of a version (this is the version of the config format, for now just ``v1``), ``kind`` of config (in this case
``datasets``), and ``specs``. ``specs`` is where the dataset initialisation keyword arguments are defined. The yaml file can include
configs for other ``molflux`` packages as well (see [Standard API](../standard_api/intro.md#load-from-yaml) for more info).
To load this yaml file, you can simply do


```{code-block} ipython3
from molflux.datasets import load_from_yaml

datasets = load_from_yaml(path_to_yaml_file)
```


## Working with datasets

``molflux.datasets`` was designed to supplement the HuggingFace [datasets](https://huggingface.co/docs/datasets/index) library,
giving you access to our additional catalogue of datasets and to a number of convenient utility functions. The datasets
returned by e.g. `molflux.datasets.load_dataset()` are actually native HuggingFace datasets, with all of the associated
functionality.

You can find complete documentation on how to work with HuggingFace datasets [online](https://huggingface.co/docs/datasets/index), or
check out their official [training course](https://huggingface.co/course/chapter5/1?fw=pt)! The rest of this
tutorial will show a couple of examples of some of the most basic functionalities available.

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

dataset.to_pandas()
```

You can also save and load the datasets to disk or to the cloud (s3)

```{code-block} python
from molflux.datasets import load_dataset, load_dataset_from_store, save_dataset_to_store

dataset = load_dataset('esol')

save_dataset_to_store(dataset, "my/data/dataset.parquet")

dataset = load_dataset_from_store("my/data/dataset.parquet")
```

```{seealso}

For more information on how to save, load (from disk), featurise, and split datasets, see these guides:
* [How to save datasets](saving.md)
* [How to load datasets](loading.md)
* [How to featurise datasets](featurising.md)
* [How to split datasets](splitting.md)
```
