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

# More data loading options


Your datasets can be stored in various places; they can be in the `molflux.datasets` catalogue, on your local machine's disk,
on a remote disk, on the HuggingFace datasets hub, and in in-memory data structures such as Arrow tables, Python dictionaries, and Pandas
DataFrames. In all of these cases, ``molflux.datasets`` can load it.

## HuggingFace Hub

Datasets are loaded from a dataset loading script that downloads and generates the dataset. However, you can also load a
dataset from any dataset repository on the HuggingFace Hub without a loading script!
You just need to use the {func}`molflux.datasets.load_dataset` function to load the dataset.

For example, try loading the files from this demo repository by providing the repository namespace and dataset name. This
dataset repository contains CSV files, and the code below loads the dataset from the CSV files:

```python
from molflux.datasets import load_dataset

dataset = load_dataset("lhoestq/demo1")
```

Some datasets may have more than one version based on Git tags, branches, or commits. Use the revision parameter to
specify the dataset version you want to load:

```python
from molflux.datasets import load_dataset

dataset = load_dataset(
    "lhoestq/custom_squad",
    revision="main"  # tag name, or branch name, or commit hash
)
```

## The MolFlux catalogue

Similarly, to load a dataset from the ``molflux`` catalogue:

```python
from molflux.datasets import load_dataset

dataset = load_dataset("esol")
```

Remember that you can have a look at the datasets available in the catalogue by doing
```{code-cell} ipython3
from molflux.datasets import list_datasets

catalogue = list_datasets()
print(catalogue)
```

## Persisted file formats

Datasets can also be loaded from local files stored on your computer or in the cloud. The datasets could be
stored as ``parquet``, ``csv``, ``json``, or ``txt`` files. The {func}`molflux.datasets.load_dataset_from_store` function can load each
of these file types

```{hint}
This will work automatically for both local and cloud data. If you need more fine-grained control over the filesystem,
you can pass your own `fsspec`-compatible filesystem object to `load_from_store()` as an argument to the `fs` parameter.

For convenience, we have also made available a custom AWS S3 Filesystem `fsspec` implementation which you can create with
`fsspec.filesystem("s3")`.
```

### CSV

You can read a dataset made up of one or several CSV files:

```python
from molflux.datasets.alexandria import load_dataset_from_store

dataset = load_dataset_from_store("my_file.csv")
```

If you are working with partitioned files, you can also load several CSV files at once:

```python
data_files = ["my_file_1.csv", "my_file_2.csv", "my_file_3.csv"]
dataset_dict = load_dataset_from_store(data_files)
```

You can also map the training and test splits to specific CSV files, and load them in as a `DatasetDict` or as a single
`Dataset`:

```python
data_files = {"train": ["my_train_file_1.csv", "my_train_file_2.csv"], "test": "my_test_file.csv"}

# load as a DatasetDict
dataset_dict = load_dataset_from_store(data_files)

# load just the 'train' split
dataset = load_dataset_from_store(data_files, split="train")

# merge all splits into a single Dataset
dataset = load_dataset_from_store(data_files, split="all")
```

To load remote CSV files via HTTP, pass the URLs instead:

```python
base_url = "https://huggingface.co/datasets/lhoestq/demo1/resolve/main/data/"
data_files = {'train': base_url + 'train.csv', 'test': base_url + 'test.csv'}
dataset_dict = load_dataset_from_store(data_files)
```

To load zipped CSV files you might need to explicitly provide the persistence format:

```python
data_files = "data.zip"
dataset = load_dataset_from_store(data_files, format="csv")
```

### Parquet

Parquet files are stored in a columnar format, unlike row-based files like a CSV.
Large datasets may be stored in a Parquet file because it is more efficient and faster at returning your query.

You can load Parquet files in the same way as the CSV examples shown above. For example, to load a Parquet file:

```python
from molflux.datasets import load_dataset_from_store

dataset = load_dataset_from_store("my_file.parquet")
```

You can also map the training and test splits to specific Parquet files:

```python
from molflux.datasets import load_dataset_from_store

data_files = {'train': 'train.parquet', 'test': 'test.parquet'}
dataset_dict = load_dataset_from_store(data_files)
```

To load remote Parquet files via HTTP, pass the URLs instead:

```python
base_url = "https://storage.googleapis.com/huggingface-nlp/cache/datasets/wikipedia/20200501.en/1.0.0/"
data_files = {"train": base_url + "wikipedia-train.parquet"}
wiki = load_dataset_from_store(data_files, split="train")
```

### JSON

JSON files are loaded directly as shown below:

```python
from molflux.datasets import load_dataset_from_store

dataset = load_dataset_from_store("my_file.json")
```

JSON files have diverse formats, but we think the most efficient format is to have multiple JSON objects; each line represents
an individual row of data. For example:

```json lines
{"a": 1, "b": 2.0, "c": "foo", "d": false}
{"a": 4, "b": -5.5, "c": null, "d": true}
```

Another JSON format you may encounter is a nested field, in which case you'll need to specify the field argument as shown in the following:

```json
{"version": "0.1.0",
 "data": [{"a": 1, "b": 2.0, "c": "foo", "d": false},
          {"a": 4, "b": -5.5, "c": null, "d": true}]
}
```

```python
dataset = load_dataset_from_store("my_file.json", field="data")
```

To load remote JSON files via HTTP, pass the URLs instead:

```python
base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
data_files = {"train": base_url + "train-v1.1.json", "validation": base_url + "dev-v1.1.json"}
dataset = load_dataset_from_store(data_files, field="data")
```

### Disk

You can read datasets that you have previously saved with `molflux.datasets.save_dataset_to_store(..., format="disk")`.
You just need to provide the path to the directory holding your `.arrow` file(s), and the dataset's `json` metadata:

```python
from molflux.datasets import load_dataset_from_store

dataset = load_dataset_from_store("my/dataset/dir")
```

## In-memory data

To create a datasets directly from in-memory data structures like Arrow Tables, Python dictionaries and Pandas DataFrames
you can use directly HuggingFace's `datasets.Dataset` and `datasets.DatasetDict` [class methods](https://huggingface.co/docs/datasets/loading#inmemory-data).
