# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
---------------------------------------------------------

## [Unreleased]

---------------------------------------------------------
## [0.7.0] - 2024-09-05

## Added

- Added the TDC ADMET benchmarks dataset (with optional dependency `pip install molflux[tdc]`)
- Turned on `pystan` tests

## [0.6.0] - 2024-08-23

## Removed

- removed support for python 3.8 and 3.9

## Changed

- Enables `uv` and increases the use of `ruff` throughout the codebase.
- Added `standard_deviations` input argument for all uncertainty metrics except `uncertainty_based_rejection`
- Removed the `uncertainty_based_rejection` metric from the `uncertainty` suite
- The typing of `featurise_dataset` now confirms that it can act on `DatasetDict` too.
- Load backend representations from featurisation metadata using stricter unpacking in order to not trigger `UserWarning`s

## Added

- updated to use `uv`.
- Warning if `load_from_dict` is passed a dictionary with arbitrary keys outside of the expected specification that are ignored when loading a representation.
- Added `map_light` features, which are a combination of Morgan, Avalon, Reduced Graph and handcrafted descriptors from `rdkit`.

- Added uncertainty support (`predict_with_std`, `predict_with_prediction_interval`, and `sample`) for the `ensemble_regressor` model.
- Added an `average_features_regressor` model that predicts based on the average of the input model features
- Added `GammaConformityScore` and `ResidualNormalisedScore` to `mapie_regressor`. These should allow for more adaptive prediction intervals

- Added `out_of_sample_r2`regression  metric

## Fixed

- Lightning logger config sometimes required an explicit `config` field to be recognised as a logger config; this is no longer the case.

## Security

- Load torch models with `weights_only` parameter set to `True` to address potential security concerns

## [0.5.0] - 2024-07-11

## Added
- Enable multi-column representations
- Add `linear_split_with_rotation` splitting strategy
- Added a Bayesian ordinal regression model (`ordinal_classifier`).
- `mapie_regressor` now has `predict_with_std` and `sample` methods implemented based on a Gaussian approximation for the prediction interval.
- Added `calibration_gap` metric
- Added option for masking inputs by the references
- v2 featurisation metadata with support for multi-column inputs

## Fixed
- Fixed the dict for matching modules in lightning. Allows many to one matching.
- `model_config` is now correctly overridden in LightningModules. Previously a stale config could have been used.
- Release PyTorch upper bound (previously <2.1).

## Changed
- Compatible with Pydantic v1 & v2
- Lower pin on `botocore` / `boto3` to help dependency resolution when installed alongside `dvc-s3`
- Use `class_resolver` to simplify and generalise modularity inside Lightning models.
- `model.train` will now always accept a `validation_data` kwarg. If the underlying model implementation doesn't have `validation_data` in its `model._train` (or `model._train_multi_data`), it will be dropped with a warning.
- Tag format for wrapped models (`ensemble_regressor`, `ensemble_classifier`, `mapie_regressor`, `sklearn_pipeline_regressor`, `sklearn_pipeline_classifier`) changed to make clearer which base models are included. The new tag format is of the form `'{model.tag}[{base_model.tag}]'`.
- Changed behaviour of Gaussian NLL from summing likelihoods to averaging them

## Removed
- sd parser
- Deprecate usage of `mean_squared_error` with `root=True` 

## [0.4.0] - 2024-06-03

## Removed

- Drop parameter `multi_class` and `n_jobs` for `logistic_regressor` in anticipation of `numpy>=1.7` removal

## [0.3.0] - 2024-02-09

## Changed

- Upgraded `datasets>=2.17.0` which fixes a problem with flattening indices
- Removed failure tests for flattening indices


## [0.2.0] - 2024-01-31

## Changed

- Updated the `spice` dataset from 1.1.1 to 1.1.4

## Fixed

- Patch bug with multiproc and Sequence features of fixed length

## [0.1.0] - 2023-01-29

## Added

* Added `atom_pair` from `rdkit`
* Added `topological_torsion` from `rdkit`
* Added `CovarianceMixin` for `modelzoo`
* Added separate `root_mean_squared_error` metric

## Changed
 
* `prediction_internal_coverage` from `numpy`
* Updated `mapie_regressor`
* Strict warnings
* removed `pkg_resources` for `importlib`
* HF `datasets` uses `trust_remote_code=True` by default
* updated `ruff~=0.1.0`
* updated `datasets>=2.16.0`

## Fixed

* Fixed `accuracy` metric 

## Removed

* Removed `pytest-lazy-fixture`  


## [0.0.1] - 2023-12-15

## Removed

* Removed a featuriser

## [0.0.0] - 2023-12-15

## Added

* Initial release

---------------------------------------------------------

## [X.Y.Z] - 20YY-MM-DD

(Template)

### Added

For new features.

### Changed

For changes in existing functionality.

### Deprecated

For soon-to-be removed features.

### Removed

For now removed features.

### Fixed

For any bug fixes.

### Security

In case of vulnerabilities.
