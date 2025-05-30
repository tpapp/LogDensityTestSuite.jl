# Changelog

## Unreleased

### Changed

- transformations now act as functions, eg `linear(A, ℓ)` should be written as `linear(A)(ℓ)`
- `linear` no longer takes a dimensionless matrix-like quantity like `UniformScaling`
- a `DimensionError` is thrown instead of an `ArgumentError` when a transformation is not applicable to a distribution

### Added

- `funnel`

### Added

## 0.6.6

Sorry, a changelog was not kept for this phase of the package as it was very experimental.
