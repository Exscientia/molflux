# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
---------------------------------------------------------

## [Unreleased]

---------------------------------------------------------

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
