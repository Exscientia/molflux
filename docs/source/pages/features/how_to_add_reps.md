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

# Add your own representations

Even though ``molflux.features`` comes with built-in representations, you may want to add your own feature extractor.
In this guide, we will go through two ways you can do this.

## Temporary representations

The first method is by registering your representation temporarily. This is useful in cases when you are still prototyping
and would like to have easy access to your representation code and modify it.

For a representation to be available in ``molflux.features``, it must have a representation class. The basic skeleton looks like this

```{code-cell} ipython3
from typing import Any, Dict, List

from molflux.features.catalogue import register_representation
from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.typing import MolArray


_DESCRIPTION = """
My new representation.
"""

@register_representation(kind = "custom", name = "your_rep_name")
class YourRepName(RepresentationBase):

    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        """ compute your features!"""

        my_computed_features = ["bla, bla, bla"]
        feature_key = self.tag

        return {feature_key: my_computed_features}


from molflux.features import load_representation

rep = load_representation(name="your_rep_name")

print(rep)
```

Let's break this down.

Start by creating a class (you can name it whatever you like, the class name has no impact on
how the representation will appear in ``molflux.features``). This class must inherit ``RepresentationBase`` from ``molflux.features``.

Next, since we are adding this representation temporarily, you need to register it (basically make it available in ``molflux.features``) by
adding the decorator ``register_representation`` to the class. Here, you specify the kind and ``name`` of your representation.
This is how it will appear in the ``molflux.features`` catalogue and ``name`` is how it will be loaded.

Next, you should also add a description to your representation. This is done by adding an ``_info`` method which returns
a ``RepresentationInfo`` object.

Finally, you need to implement the ``_featurise`` method. This is the method used to compute the features. There are
two requirements:

- **Signature**: The ``_featurise`` method must have the following arguments
  - A ``samples`` argument of type ``MolArray``. This is the list of molecules to be featurised.
  - After that, you can add all the optional arguments for featurisation **with** default values.
  - Finally, you need to have ``**kwargs: Any`` (this is for mypy type checking reasons).
- **Return type**: The ``_featurise`` method must return a dictionary of the computed features with string identifiers
as keys (can be anything but it is recommended that you use ``self.tag``, the representation tag which is automatically generated)
and the features as values. You can return multiple features from the same representation as a dictionary with multiple key, value pairs.

Et Voilà ! As you can see from the python cell above, the representation was loaded using ``molflux.features``!


## Permanent representations

If you have settled on a representation and would like to have it as part of a package, you can still use the above method
to do this (add a script to your repo with the above code). If you do this, your representation will only be available
in ``molflux.features`` if you have imported your package (and also imported the representation class from your package).

To avoid this, we strongly recommended that you add your representation as a plugin (to learn more about plugins see
[here](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)). Using this method, your representation
will appear and be available in ``molflux.features`` automatically (you need to have it pip installed in the same environment).

To begin with, add your representation class to your repo (this can be any repo, not necessarily ``molflux.features``). Next, remove the
``register_representation`` decorator (it's no longer needed).

```{code-cell} ipython3
from typing import Any, Dict, List

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.typing import MolArray


_DESCRIPTION = """
My new representation.
"""

class YourRepName(RepresentationBase):

    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        """ compute your features!"""

        my_computed_features = ["bla, bla, bla"]
        feature_key = self.tag

        return {feature_key: my_computed_features}
```

Now, go to the ``pyproject.toml`` file of your repo and under ``[project.entry-points.'molflux.features.plugins.pluging_name']``,
add a plugin to your representation class as follows

```{code-block} ini
[project.entry-points.'molflux.features.plugins.plugin_name']
name_of_representation = 'path.to.module.file:YourRepName'
```

```{note}
You can also do this in the ``setup.cfg`` file of your repo and under ``[options.entry_points]``. Add a plugin to your
representation class as follows

```{code-block} ini
[options.entry_points]
molflux.features.plugins.kind_of_representation =
   name_of_representation = path.to.module.file:YourRepName
```

Let's break this down. This entry point allows ``molflux.features`` to detect your representation (you need to have both
``molflux`` and your package installed in the same environment, the order of installation does not matter). In the plugin definition,
you can specify the kind of your representation. Next, you specify the path to the class of your representation. Here you
define the name of your representation.

And that's it! You can now find your representation available in ``molflux.features``.
