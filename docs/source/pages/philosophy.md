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

MolFlux is a foundational package for molecular predictive modelling. It covers the five main pillars of molecular machine learning

- 🗂️ Datasets: Accessing and handling data
- 🌈 Features: Computing and extracting features
- 🔀 Splits: Partitioning data for model training and benchmarking
- 🤖 Modelzoo: Handling and training models
- 📈 Metrics: Computing metrics for a variety of tasks


MolFlux provides self-contained access to the machine learning ecosystem to enable you to build
machine learning models from scratch.

## The Standard API

One of the main challenges of building machine learning models and keeping up to date with the state-of-the-art is the
variety of APIs and interfaces that different models and model packages follow. Even the same submodules in the same
package can have different APIs. This makes the work of using and comparing the rapidly increasing number of models and
features difficult and time-consuming.

The unifying principle of MolFlux is standardisation. Whether you're trying to extract basic features from data, use a
simple random forest regressor, or trying to train a complicated neural network, the API is the same. The motto is "if
you learn it once, you know it always"! What the standard API also provides is smooth interaction between the different
submodules.

## Modular

Including so much functionality in one package is not trivial and python dependencies can often become daunting. The
MolFlux package handles this by being highly modular. You can easily unlock more functionality by installing their
corresponding dependencies.

The modular design of the system greatly simplifies the integration of new models, features, and datasets. Its robust,
yet simple abstractions are capable of managing everything from basic to complex models and features.

## Acknowledgements

The ``molflux`` package has been developed by researchers and engineers at Exscientia

* Alan Bilsland
* Julia Buhmann
* Ward Haddadin
* Jonathan Harrison
* Dom Miketa
* Emil Nichita
* Stefanie Speichert
* Hagen Triendl
* Alvise Vianello
