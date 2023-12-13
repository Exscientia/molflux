# Splits in a nutshell

{sub-ref}`wordcount-words` words | {sub-ref}`wordcount-minutes` min read

One of the core steps of machine learning is splitting your data for training, validation, and testing. Splitting strategies
for drug discovery in particular are numerous and rapidly evolving. New tools and techniques are constantly being developed
but using multiple tools from different sources quickly becomes tedious, inconvenient, and prone to incompatibilities.

The ``splits`` submodule aims to address these issues. It is a collection of many different types
of splitters which can split datasets using a variety of criteria. Whether you are looking for random splits or
more complicated scaffold or time splits, ``splits`` provides a standard and modular interface for using these splitters
and also allows you to add your own!
