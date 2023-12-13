# Metrics in a nutshell


{sub-ref}`wordcount-words` words | {sub-ref}`wordcount-minutes` min read

One of the pillars of machine learning is measuring the performance of models. New tools and techniques to do this are
constantly being developed but using multiple tools from different sources can quickly become tedious, inconvenient, and
prone to incompatibilities.

The ``metrics`` submodule aims to address these issues. It is a collection of many different types of metrics
which can be computed on model predictions to evaluate the performance. Whether you are trying to compute regression metrics
(such as ``r2``) or classification ones, ``metrics`` provides a standard and modular interface for using these
metrics and also allows you to add your own!
