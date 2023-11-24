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

# What is MolFlux?

## Background

MolFlux is a foundational package for molecular predictive modelling. It covers the five main pillars of machine learning

1) Datasets: accessing and handling data
2) Features: computing and extracting features
3) Splits: partitioning data for model training and benchmarking
4) Modelzoo: access and handling of models
5) Metrics: computing metrics for a variety of tasks

Molflux provides complete and self-contained access to the whole machine learning ecosystem to enable you to build
machine learning models from scratch.

## The Standard API

One of the main challenges of building machine learning models and keeping up to date with the state-of-the-art is the
variety of APIs and interfaces that different models and model packages follow. Even the same submodules in the same
package can have different APIs. This makes the work of using and comparing the rapidly increasing models and features
difficult and time consuming. What the standard API also provides is smooth interaction between the different submodules.

The unifying principle of MolFlux is standardisation. Whether you're trying to extract basic features from data, use a
simple random forest regressor, or trying to train a complicated neural network, the API is the same. The motto is "if
you learn it once, you know it always"!

## Modular

Including so much functionality in one package is not trivial and python dependencies can often become overbearing. The
MolFlux package handles this by being highly modular. All one needs to do to get access to more functionality is to install
the relevant dependencies.

The modular nature also makes adding new models, features, and datasets much easier. The robust, but simple, abstractions
can handle simple models and features to complicated ones.
