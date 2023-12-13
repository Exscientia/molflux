# Features in a nutshell

{sub-ref}`wordcount-words` words | {sub-ref}`wordcount-minutes` min read

One of the pillars of machine learning is extracting useful features for models to learn with. Machine learning features
for drug discovery in particular are numerous and rapidly evolving. New techniques are constantly emerging but using
tools from different sources quickly becomes tedious, inconvenient, and prone to incompatibilities.

The ``features`` submodule aims to address these issues. It is a collection of many different
types of featurisers which transform molecules (or chemical structures in general) into a spectrum of machine learning
features. Whether you are looking for engineered features (such as fingerprints) or learned features (from neural network
embeddings), ``features`` provides a standard and modular interface for using these featurisers and also allows you to
add your own!
