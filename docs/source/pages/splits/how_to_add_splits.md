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

# Add your own splitting strategy

Even though ``molflux.splits`` comes with built-in splitting strategies, you may want to add your own splitter. In this guide,
we go through two ways you can do this.

## Temporary splitting strategy

The first method is by registering your strategy temporarily. This is useful in cases when you are still prototyping
and would like to have easy access to your strategy code and modify it.

For a splitting strategy to be available in ``molflux.splits``, it must have a splitting strategy class. The basic skeleton looks like this

```{code-cell} ipython3
from typing import Any, Iterator, Optional

from molflux.splits.catalogue import register_splitting_strategy
from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

_DESCRIPTION = """
My new splitting strategy
"""


@register_splitting_strategy(kind = "custom", name = "your_split_name")
class YourNewSplit(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:


        # my new splitting strategy
        train_indices = [0, 3, 5]
        validation_indices = [1, 2]
        test_indices = [4]

        yield train_indices, validation_indices, test_indices

from molflux.splits import load_splitting_strategy

strategy = load_splitting_strategy(name='your_split_name')

print(strategy)
```

Let's break this down.

Start by creating a class (you can name it whatever you like, the class name has no impact on
how the strategy will appear in ``molflux.splits``). This class must inherit ``SplittingStrategyBase`` from ``molflux.splits``.

Next, since we are adding this splitting strategy temporarily, you need to register it (basically make it available in ``molflux.splits``) by
adding the decorator ``register_splitting_strategy`` to the class. Here, you specify the kind and ``name`` of your strategy.
This is how it will appear in the ``molflux.splits`` catalogue and ``name`` is how it will be loaded.

Next, you should also add a description to your strategy. This is done by adding an ``_info`` method which returns
a ``SplittingStrategyInfo`` object.

Finally, you need to implement the ``_split`` method. This is the method used to compute the split indices. There are
two requirements:

- **Signature**: The ``_split`` method must have the following arguments
  - A ``dataset`` argument of type ``Splittable``. This is the data to be split (``Splittable`` enforces that it has a size).
  - ``y``, the target variable for supervised learning problems. Default to ``None`` if not needed.
  - ``group``, group labels for the samples used while splitting the dataset. Default to ``None`` if not needed
  - After that, you can add all the optional arguments for splitting **with** default values.
  - Finally, you need to have ``**kwargs: Any`` (this is for mypy type checking reasons).
- **Return type**: The ``_split`` method must ``yield`` a tuple of ``(train_indices, validation_indices, test_indices)``.

Et voilà  ! As you can see from the python cell above, the strategy was loaded using ``molflux.splits``!


## Permanent splitting strategy

If you have settled on a splitting strategy and would like to have it as part of a package, you can still use the above method
to do this (add a script to your repo with the above code). If you do this though, your strategy will only be available
in ``molflux.splits`` if you have imported your package (and also imported the strategy class from your package).

To avoid this, we strongly recommended that you add your strategy as a plugin (to learn more about plugins see
[here](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)). Using this method, your strategy
will appear and be available in ``molflux.splits`` automatically (you need to have it pip installed in the same environment).

To begin with, add your strategy class to your repo (this can be any repo, not necessarily ``molflux``). Next, remove the
``register_splitting_strategy`` decorator (it's no longer needed).

```{code-cell} ipython3
from typing import Any, Iterator, Optional

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

_DESCRIPTION = """
My new splitting strategy
"""

class YourNewSplit(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:


        # my new splitting strategy
        train_indices = [0, 3, 5]
        validation_indices = [1, 2]
        test_indices = [4]

        yield train_indices, validation_indices, test_indices
```

Now, go to the ``pyproject.toml`` file of your repo and under ``[project.entry-points.'molflux.splits.plugins.pluging_name']``,
add a plugin to your strategy class as follows

```{code-block} ini
[project.entry-points.'molflux.splits.plugins.plugin_name']
name_of_strategy = 'path.to.module.file:YourNewSplit'
```

```{note}
You can also do this in the ``setup.cfg`` file of your repo and under ``[options.entry_points]``. Add a plugin to your
strategy class as follows

```{code-block} ini
[options.entry_points]
molflux.splits.plugins.kind_of_strategy =
   name_of_strategy = path.to.module.file:YourNewSplit
```

Let's break this down. This entry point allows ``molflux.splits`` to detect your strategy (you need to have both ``molflux`` and your
package installed in the same environment, the order of installation does not matter). In the plugin definition, you can
specify the kind of your strategy. Next, you specify the path to the class of your strategy. Here you
define the name of your strategy.

And that's it! You can now find your strategy available in ``molflux.splits``.
