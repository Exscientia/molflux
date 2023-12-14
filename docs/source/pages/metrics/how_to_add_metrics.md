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

# Add your own metrics

Even though ``molflux.metrics`` comes with built-in metrics, you may want to add your own awesome metric. In this guide,
we will go through two ways you can do this.

## Temporary metrics

The first method is by registering your metric temporarily. This is useful in cases when you are still prototyping
and would like to have easy access to your metric code and modify it.

For a metric to be available in ``molflux.metrics``, it must have a metric class. The basic skeleton looks like this

```{code-cell} ipython3
from typing import Any, Dict, List, Optional

import datasets

from molflux.metrics.catalogue import register_metric
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult



_DESCRIPTION = """
My new metric.
"""

@register_metric(kind = "custom", name = "your_metric_name")
class YourMetricName(HFMetric):

    def _info(self) -> datasets.MetricInfo:
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float64")),
                    "references": datasets.Sequence(datasets.Value("float64")),
                }
            )
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        **kwargs: Any,
    ) -> MetricResult:

        my_metric = 0

        return {self.tag: my_metric}

from molflux.metrics import load_metric

metric = load_metric("your_metric_name")

print(metric)

```

Let's break this down.

This specific metric is built on top of HuggingFace metrics. This is not necessary and you can override it if you wish and
build your own metric from scratch. Here, we use HuggingFace metrics because they provide a lot of convenient features.

Start by creating a class (you can name it whatever you like, the class name does not impact
how the metric will appear in ``molflux.metrics``). If you wish to use the HuggingFace features, then the class must inherit
``HFMetric`` from ``molflux.metrics``.

Next, since we are adding this metric temporarily, you need to register it (basically make it available in ``molflux.metrics``) by
adding the decorator ``register_metric`` to the class. Here, you specify the kind (under which your metric will be discovered
in the ``molflux.metrics`` catalogue) and ``name`` of your model (this is how your metric will be loaded).

Next, you should also add a description to your metric. This is done by adding an ``_info`` method which returns
a ``MetricInfo`` object.

Finally, you need to implement the ``_score`` method. This is the method used to compute the metric. There are
two requirements:

- **Signature**: The ``_metric`` method must have the following arguments
  - A ``predictions`` argument of type ``ArrayLike``. This is the list of predictions.
  - A ``references`` argument of type ``ArrayLike``. This is the list of ground truths.
  - After that, you can add all the optional arguments for computing the metric **with** default values.
  - Finally, you need to have ``**kwargs: Any`` (this is for mypy type checking reasons).
- **Return type**: The ``_score`` method must return a dictionary of the computed metric with string identifiers
as keys (can be anything but it is recommended that you use ``self.tag``, the metric tag which is automatically generated)
and the metrics as values. You can return multiple metrics from the same metric as a dictionary with multiple key, value pairs.

Et voilà  ! As you can see from the python cell above, the metric was loaded using ``molflux.metrics``!


## Permanent metrics

If you have settled on a metric and would like to have it as part of a package, you can still use the above method
to do this (add a script to your repo with the above code). If you do this, your metric will only be available
in ``molflux.metrics`` if you have imported your package (and also imported the metric class from your package).

To avoid this, we strongly recommended that you add your metric as a plugin (to learn more about plugins see
[here](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)). Using this method, your metric
will appear and be available in ``molflux.metrics`` automatically (you just need to have it pip installed in the same environment).

To begin with, add your metric class to your repo (this can be any repo, not necessarily ``molflux.metrics``). Next, get rid of the
``register_metric`` decorator (it's no longer needed).

```{code-cell} ipython3
from typing import Any, Dict, List, Optional

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult



_DESCRIPTION = """
My new metric.
"""

class YourMetricName(HFMetric):

    def _info(self) -> datasets.MetricInfo:
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation="",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("float64")),
                    "references": datasets.Sequence(datasets.Value("float64")),
                }
            )
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        **kwargs: Any,
    ) -> MetricResult:

        my_metric = 0

        return {self.tag: my_metric}
```

Now, go to the ``pyproject.toml`` file of your repo and under ``[project.entry-points.'molflux.metrics.plugins.pluging_name']``,
add a plugin to your representation class as follows

```{code-block} ini
[project.entry-points.'molflux.metrics.plugins.plugin_name']
name_of_metric = 'path.to.module.file:YourMetricName'
```

```{note}
You can also do this in the ``setup.cfg`` file of your repo and under ``[options.entry_points]``. Add a plugin to your
metric class as follows

```{code-block} ini
[options.entry_points]
molflux.metrics.plugins.kind_of_representation =
   name_of_metrics = path.to.module.file:YourMetricName
```

Let's break this down. This entry point allows ``molflux.metrics`` to detect your metric (you need to have both ``molflux.metrics`` and your
package installed in the same environment, the order of installation does not matter). In the plugin definition, you can
specify the kind of your metric. Next, you specify the path to the class of your metric. Here you
define the name of your metric.

And that's it! You can now find your metric available in ``molflux.metrics``.
