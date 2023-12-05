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


In this section, we will illustrate how to use ``molflux.splits``. These examples will provide you with a starting
point.


## Browsing

First, let's have a look at what splitting strategies are available for use. These are conveniently categorised (for example,
into ``core``, ``rdkit``, etc.). To view what's available you can do

```{code-cell} ipython3

from molflux.splits import list_splitting_strategies

catalogue = list_splitting_strategies()

print(catalogue)
```

This returns a dictionary of available splitting strategies (organised by category and ``name``). There are a few to choose from.
By default ``molflux.splits`` will come with ``core`` splitters (such as ``shuffle_split`` and ``k_fold``). You can get more
splitting strategies by pip installing extra packages (such as ``rdkit``). To see how you can add your own splitting strategy, see
[How to add your own splitting strategy](how_to_add_splits.md).

## Loading splitting strategies

Loading a ``molflux.splits`` strategy is very easy, simply do

```{code-cell} ipython3

from molflux.splits import load_splitting_strategy

strategy = load_splitting_strategy('shuffle_split')

print(strategy)
```

By printing the loaded strategy, you get more information about it. Each splitting strategy has a ``name``, and ``tag``
(to identify it). You can also see the optional splitting arguments (and their default values) in the signature.
There is also a short description of the strategy.

You can also load a splitting strategy from a config. A splitting strategy config is a dictionary specifying the strategy to be
loaded. A config dictionary must have the following format
```{code-block} python
splitting_strategy_dict = {
    'name': '<name of the strategy>',
    'config': '<kwargs for instantiating strategy>'
    'presets': '<kwarg presets for splitting>'
}
```

The ``name`` keys specify the ``name`` of the splitting strategy to load from the catalogue.
The ``config`` key specifies the arguments that are needed for instantiating the splitting strategy and
the ``presets`` key specifies preset keyword arguments to apply when splitting (for example, the train and test fractions). If neither is
specified, the splitting strategy will use default values. Loading from a config is done using the ``load_from_dict``
function.

```{code-cell} ipython3

from molflux.splits import load_from_dict
config = {
          'name': 'shuffle_split',
          'presets':
            {
              'train_fraction': 0.8,
              'validation_fraction': 0.0,
              'test_fraction': 0.2,
            }
          }

strategy = load_from_dict(config)

print(strategy.state)
```

For convenience, you can also load a group of strategies all at once by specifying a list of configs.

```{code-cell} ipython3

from molflux.splits import load_from_dicts

config = [
    {
        'name': 'shuffle_split',
        'config':
            {
                'tag': 'train_test_shuffle',
            },
        'presets':
            {
                'train_fraction': 0.8,
                'validation_fraction': 0.0,
                'test_fraction': 0.2,
            }
    },
    {
        'name': 'shuffle_split',
        'config':
            {
                'tag': 'train_val_test_shuffle',
            },
        'presets':
            {
                'train_fraction': 0.7,
                'validation_fraction': 0.2,
                'test_fraction': 0.1,
            }
    }
]

strategies = load_from_dicts(config)

print(strategies)
```

Finally, you can load strategies from a yaml file. You can use a single yaml file which includes configs for all the ``molflux`` tools
and ``molflux.splits`` will know how to extract the relevant document it needs. To do so, you need to define a yaml file with the
following example document

```{code-block} yaml

---
version: v1
kind: splits
specs:
    - name: k_fold
      presets:
          n_splits: 5

...
```

It consists of a version (this is the version of the config format, for now just ``v1``), ``kind`` of config (in this case
``splits``), and ``specs``. ``specs`` is where the configs are defined. The yaml file can include
configs for other ``molflux`` modules as well. To load this yaml file, you can simply do


```{code-block} ipython3

from molflux.splits import load_from_yaml

strategies = load_from_yaml(path_to_yaml_file)

print(strategies)
```


## Splitting

After loading a splitting strategy, you can apply it to any array-like object to get the split indices.

```{code-cell} ipython3

from molflux.splits import load_splitting_strategy

strategy = load_splitting_strategy('shuffle_split')

folds = strategy.split(range(100))

for train_indices, validation_indices, test_indices in folds:
    print(f"TRAIN: ", train_indices)
    print(f"VALIDATION: ", validation_indices)
    print(f"TEST: ", test_indices)
```

The `.split()` method will return a generator of split indices. Every time you iterate the generator, you get a tuple of
split indices (in the case of ``shuffle_split``, there is only one tuple, but other strategies such as ``k_fold`` will yield
``k`` tuples).

## Integration with ``molflux.datasets``

 You can easily split your datasets from ``molflux.datasets`` using ``molflux.splits`` splitting strategies.
 To learn more, see [here](../datasets/splitting.md).
