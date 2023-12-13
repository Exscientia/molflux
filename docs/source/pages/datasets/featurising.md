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

# Featurising

The ``molflux`` modules are built to plug into each other seamlessly. If you would like to featurise your datasets
with ``molflux.features`` representations (or any other representation following the same API), you can easily do so as follows:

```python
from molflux.datasets import load_dataset, featurise_dataset

dataset = load_dataset('esol')

# representations = <your molflux.features representations>

featurised_dataset = featurise_dataset.featurise_dataset(
    dataset=dataset,
    column="<column to be featurised>",
    representations=representations
)
```

This returns a new datasets with the required features as new columns (if you use multiple representations all at once using
the ``load_from_dicts`` method of ``molflux.features``, then they will each create a new column with their computed features).

Under the hood, this is done using the ``map`` functionality of HuggingFace datasets. You can pass some ``kwargs`` to control
the featurisation. The full set of ``kwargs`` can be found [here](https://huggingface.co/docs/datasets/v2.3.2/en/package_reference/main_classes#datasets.Dataset.map)
but the most useful ones are
- ``batch_size Optional[int] = 1000``: the size of the batches.
- ``num_proc: Optional[int] = None``: maximum number of processes for featurisation.

```{seealso}
There is also a complete workflow example that also covers how datasets and featurization are integrated: [ESOL Training](../tutorials/esol_reg.md#featurising).
```

## Tweaking the featurised columns' names

By default, the featurised columns names will encode information both about the feature name and the name of the source
column that was featurised. This allows you to keep track of how your dataset columns have been featurised over time,
and provides uniquely identifiable column names even for columns featurised by the same representations.

If needed, you can also assign custom display names through the `display_names` argument, which should be a nested list
of display names for each representation that you are applying:

```{code-cell} ipython3
from molflux.datasets import load_dataset, featurise_dataset
from molflux.features import load_from_dicts as load_reps_from_dicts

dataset = load_dataset('esol')

representations = load_reps_from_dicts(
    [
        {"name": "morgan"},
        {"name": "character_count"},
        {"name": "maccs_rdkit"},
    ]
)

display_names = [["my_morgan_fingerprint"], ["my_character_count"], [None]]

featurised_dataset = featurise_dataset(
    dataset=dataset,
    column="smiles",
    representations=representations,
    display_names=display_names
)

print(featurised_dataset.column_names)
```

where `None` can be used as a placeholder for features for which you don't need to set a custom display name (a
custom naming template will be applied).

The `display_names` argument can also accept a templated string that will be dynamically injected with context available
at runtime. This is useful if you would like the datasets to be featurised according to a specific formatting convention:

```{code-cell} ipython3
display_names = "{source_column}>>{feature_name}"

featurised_dataset = featurise_dataset(
    dataset=dataset,
    column="smiles",
    representations=representations,
    display_names=display_names
)

print(featurised_dataset.column_names)
```

For the time being, `source_column` and `feature_name` are the only keys that can be requested from the context.

You can also mix and match the features shown above:

```{code-cell} ipython3
display_names = [["my_circular_fingerprint"], ["{source_column}>>{feature_name}"], [None]]

featurised_dataset = featurise_dataset(
    dataset=dataset,
    column="smiles",
    representations=representations,
    display_names=display_names
)

print(featurised_dataset.column_names)
```
