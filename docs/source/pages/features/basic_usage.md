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


In this section, we illustrate how to use ``molflux.features``. These examples will provide you with a starting
point.


## Browsing

First, we review what representations are available for use. These are conveniently categorised (for example,
into ``core``, ``rdkit``, etc.). To view what's available you can do

```{code-cell} ipython3

from molflux.features import list_representations

catalogue = list_representations()

print(catalogue)
```

This returns a dictionary of available representations (organised by categories and ``name``). There are a few to choose from.
By default ``molflux.features`` will come with ``core`` features. You can get more representations by pip installing packages
which have ``molflux`` representations. To see how you can add your own representation, see [How to add your own representations](how_to_add_reps.md).

## Loading representations

Loading a ``molflux.features`` representation is very easy, simply do

```{code-cell} ipython3

from molflux.features import load_representation

representation = load_representation(name="morgan")

print(representation)
```

By printing the loaded representation, you get more information about it. Each representation has a ``name``, and a ``tag``
(to uniquely identify it in case you would like to generate multiple copies of the same representations but with different
configurations). You can also see the optional featurisation arguments (and their default values) in the signature.
There is also a short description of the representation.

You can also load a representation from a config. A ``molflux.features`` config is a dictionary specifying the representation
to be loaded. A config dictionary must have the following format
```{code-block} python
representation_dict = {
    'name': '<name of the representation>',
    'config': '<kwargs for instantiating representation>'
    'presets': '<kwarg presets for featurising>'
}
```

The ``name`` key specifies the ``name``  of the representation to load from the catalogue. The ``config`` key
specifies the arguments that are needed for instantiating the representation and the ``presets`` key specifies some preset
kwargs to apply upon featurisation (for example, the length of a fingerprint). If neither is specified, the
representation will use default values.

To load a representation from a config

```{code-cell} ipython3

from molflux.features import load_from_dict

config = {
    'name': 'morgan',
    'presets':
        {
            'n_bits': 16,
            'radius': 3,
        },
}

representation = load_from_dict(config)

print(representation.state)
```

For convenience, you can also load a group of representations all at once by specifying a list of configs.

```{code-cell} ipython3

from molflux.features import load_from_dicts

config = [
        {
            'name': 'character_count',
        },
        {
            'name': 'morgan',
            'presets':
                {
                    'n_bits': 16,
                    'radius': 4,
                },
        }
]

representations = load_from_dicts(config)

print(representations)
```

Finally, you can load representations from a yaml file. You can use a single yaml file which includes configs for all the ``molflux`` tools,
and ``molflux.features`` will know how to extract the relevant document it needs. To do so, you need to define a yaml file with the
following example document:

```{code-block} yaml

---
version: v1
kind: representations
specs:
    - name: character_count
    - name: morgan
      presets:
        - n_bits: 16
        - radius: 4
...
```

It consists of a version (this is the version of the config format, for now just ``v1``), ``kind`` of config (in this case
``representations``), and ``specs``. ``specs`` is where the configs are defined. The yaml file can include
configs for other ``molflux`` modules as well. To load this yaml file, you can simply do


```{code-block} ipython3

from molflux.features import load_from_yaml

representations = load_from_yaml(path_to_yaml_file)

print(representations)
```


## Featurisation

After loading a representation (or group of representations), you can apply them to molecules to compute the features.
The input to ``molflux.features`` depends on the representation, but in general all representations can accept ``SMILES``
(or binary serialised molecules from ``rdkit`` or ``openeye``). Molecules can be passed individually or as a list.

```{code-cell} ipython3

from molflux.features import load_representation

representation = load_representation("character_count")

data = ["CCCC", "c1ccc(cc1)C(C#N)OC2C(C(C(C(O2)COC3C(C(C(C(O3)CO)O)O)O)O)O)O"]
featurised_data = representation.featurise(data)

print(featurised_data)
```

This will return a dictionary with the representation ``tag`` as the key and the computed features as the value. For a group
of representations, you can follow the same procedure

```{code-cell} ipython3

from molflux.features import load_from_dicts

feature_config = [
        {
            'name': 'character_count',
        },
        {
            'name': 'morgan',
            'presets':
                {
                    'n_bits': 16,
                    'radius': 4,
                },
        }
    ]

representations = load_from_dicts(feature_config)

data = ["CCCC", "c1ccc(cc1)C(C#N)OC2C(C(C(C(O2)COC3C(C(C(C(O3)CO)O)O)O)O)O)O"]
featurised_data = representations.featurise(data)

print(featurised_data)
```

This will return a dictionary with all the features (where the ``tags`` as the keys and the features as the values).

```{note}
The ``molflux`` package also builds on top of the above featurising methods to reproduce featurisation in production.
See [Productionising featurisation](../production/featurisation.md).
```


## Integration with ``molflux.datasets``

 You can easily featurise your datasets from ``molflux.datasets`` using ``molflux.features`` representations.
 To learn more, see [here](../datasets/featurising.md).
