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

# Splitting

The ``molflux`` modules are built to plug into each other seamlessly. If you would like to split your datasets
with splitting strategies from ``molflux.splits`` (or any other splitting strategies following the same API), you can easily
do so as follows

```python
from molflux.datasets import load_dataset, split_dataset

dataset = load_dataset("esol")

# splitting_strategy = <your splitting strategy>

folds = split_dataset(dataset, strategy=splitting_strategy)
```

This returns a generator of folds. A fold is a `datasets.DatasetDict` dictionary of datasets with the split names as
keys and `datasets.Dataset` datasets as values. To generate each fold, just iterate through the generator or manually
yield from the generator using ``next``.

In practice, the following example should get you started:

```{code-cell} ipython3
from molflux.datasets import load_dataset, split_dataset
from molflux.splits import load_splitting_strategy

dataset = load_dataset("esol")
splitting_strategy =  load_splitting_strategy("k_fold")

folds = split_dataset(dataset, strategy=splitting_strategy)

for fold in folds:
    # do anything you want!
    print(fold)
```

```{seealso}
There is also a complete workflow example that also covers how datasets and splitting are integrated: [ESOL Training](../tutorials/esol_reg.md#splitting).
```
