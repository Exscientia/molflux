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

# Save a dataset


Your data can be stored in various places; on your local machine's disk, or as in in-memory data structures like Arrow
tables, Python dictionaries and Pandas DataFrames. This guide will show you how to do this.

## Persisted file formats

Datasets (`Dataset` and `DatasetDict`) can be stored as local files on your computer, or in the cloud. The datasets
could be stored as a parquet, csv, or json file. The {func}`molflux.datasets.save_dataset_to_store` function can save
your datasets as each of these file types.

```{hint}

This will work automatically for both local and cloud data. If you need more fine-grained control over the filesystem,
you can pass your own `fsspec`-compatible filesystem object to `load_dataset_from_store()` as an argument to the `fs`
parameter.

For convenience, we also make available a custom AWS S3 Filesystem `fsspec` implementation which you can create with
`fsspec.filesystem("s3")`.
```

### Parquet

Parquet files are stored in a columnar format, unlike row-based files like a CSV.
Large datasets may be stored in a Parquet file because it is more efficient and faster at returning your query.

To save a dataset to Parquet:

```python
from molflux.datasets import save_dataset_to_store

save_dataset_to_store(dataset, path="s3://my-bucket/my_file.parquet")
```

You can also save `DatasetDicts`. In this case, the target path should point at a directory where the
individual splits will be saved.

```python
from molflux.datasets import save_dataset_to_store

save_dataset_to_store(dataset_dict, path="s3://my-bucket/data")
```

For other persistence formats, a `format` will need to be specified to tell `molflux.datasets` which file format the
`DataseDict` should be saved as.

### CSV

You can store your dataset as CSV:

```python
from molflux.datasets import save_dataset_to_store

# save a Dataset
save_dataset_to_store(dataset, path="my_file.csv")

# save a DatasetDict
save_dataset_to_store(dataset_dict, path="my/data", format="csv")
```

### JSON

You can store your dataset as JSON as shown below:

```python
from molflux.datasets import save_dataset_to_store

# save a Dataset
save_dataset_to_store(dataset, path="my_file.json")

# save a DatasetDict
save_dataset_to_store(dataset_dict, path="my/data", format="json")
```


### Disk

You can store your dataset on disk as a collection of  `.arrow` file(s) and the dataset's `json` metadata:

```python
from molflux.datasets import save_dataset_to_store

# save a Dataset
save_dataset_to_store(dataset, path="my/dataset/dir")

# save a DatasetDict
save_dataset_to_store(dataset_dict, path="my/data", format="disk")
```
