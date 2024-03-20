# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/) 
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.7.2]

### Fixed
- Changed relative link to the watermasking readme in the repo readme to the full URL, so that the link is valid when readme content is mirrored in hyp3-docs

## [0.7.1]

### Added
- A description of the `asf_tools.watermasking` sub-package has been added to the [`asf_tools` README](src/asf_tools/README.md)
- Installation instructions for `osmium-tool` have been added to the [`asf_tools.watermasking` README](src/asf_tools/watermasking/README.md)

### Fixed
- `osmium-tool` dependency handling. Because `osmium-tool` is not distributed on PyPI and thus is not installed when `pip` installing `asf_tools`, `asf_tools` will now raise an `ImportError` when `osmium-tool` is missing that provides installation instructions. Note: `osmium-tool` is distributed on conda-forge and will be included when conda installing `asf_tools`.

## [0.7.0]

### Added
* Scripts and entrypoints for generating our global watermasking dataset added to `watermasking`.

## [0.6.0]

### Added
* You can choose whether the `ts` (threat score; default) or `fmi` (Fowlkes-Mallows index) minimization metric is used for the flood mapping iterative estimator:
  * the `flood_map` console script entrypoint now accepts a `--minimization-metric` argument
  * the  `asf_tools.hydrosar.floopd_map.make_flood_map` function now accepts a `minimization_metric` keyword argument
* The flood mapping iterative estimator will ignore waterbodies smaller than a minimum number of pixels (default = 0)
  * the `flood_map` console script entrypoint now accepts a `--iterative-min-size` argument
  * the  `asf_tools.hydrosar.floopd_map.make_flood_map` function now accepts a `iterative_min_size` keyword argument

### Changed
* The HydroSAR code (`flood_map`, `water_map`, and `hand`) in `asf_tools` has been isolated to an `asf_tools.hydrosar` sub-package
* The `asf_tools.hydrosar.flood_map.iterative` estimator now runs with a maximum step size of 3 instead of the default 0.5.
* The `asf_tools.hydrosar.flood_map.iterative` estimator now uses the mean of the iterative bounds at the initial guess.
* the known water threshold used to determine perennial water when creating flood maps will be calculated `asf_tools.hydrosar.flood_map.get_pw_threshold` if not provided
* `get_epsg_code` and `epsg_to_wkt` have been moved from`asf_tools.composite` to `asf_tools.util`
* `read_as_array` and `write_cog` have been moved from`asf_tools.composite` to `asf_tools.raster`
* `get_coordinates` has been moved from`asf_tools.flood_map` to `asf_tools.util`

### Deprecated
* The `asf_tools.hydrosar` sub-package is being moved to the [HydroSAR project repository](https://github.com/fjmeyer/hydrosar) and will be provided in a new pip/conda installable package `hydrosar`. The `asf_tools.hydrosar` subpackage will be removed in a future release.

### Fixed
* Reverted the special handling of nan values introduced in v0.5.2, now that GDAL v3.7.0 has been released.

## [0.5.2]

### Added
* Updated tests to use a water_map/flood_map job that was created using a known input SLC

### Fixed
* Patched issue with gdalcompare.py's handling of nan values by allowing one differences between two rasters
  that contain nan values. This patch can be remove once the upstream fix is released within GDAL (likely v3.7.0)
* Fixed incorrect datatype being set for `flood_mask` GeoTIFFs leading to missing nodata.

## [0.5.1]

### Changed
* `asf_tools.flood_map` now produces rasters with pixel values of a positive integer where water is present and `0` where water is not present. Everywhere else is set to nodata.

## [0.5.0]

### Added
* HyP3 plugin entrypoints `water_map` and `flood_depth`
  * Added fuzzy and initial VV and VH geotiffs back to water map output package.
* `asf_tools.__main__` entrypoint that allows you to select which hyp3 plugin entrypoint you'd like to run 
  (e.g., `python -m asf_tools ++process water_map ...`)
  
### Changed
* `src/asf_tools/etc/entrypoint.sh` is now the docker container entrypoint, which is a simple wrapper script around 
  `python -m asf_tools`
* Temporary `numpy` version pin was removed; see [#160](https://github.com/ASFHyP3/asf-tools/pull/160)

## [0.4.6]

### Changed
* Updated the metadata for the RGB Decomposition tool in the ArcGIS Toolbox to more accurately reflect behavior
* Minor formatting and content corrections in all ArcGIS Toolbox xml files

## [0.4.5]

### Changed
* `asf-tools` now uses a `src` layout per this [recommendation](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).
* `asf-tools` now only uses `pyproject.toml` for package creation now that `setuptools` recommends [not using setup.py](https://setuptools.pypa.io/en/latest/userguide/quickstart.html#setuppy-discouraged).
* Temporarily pin `numpy` to `<1.2.4`; See: [#160](https://github.com/ASFHyP3/asf-tools/pull/160)```

## [0.4.4]

### Changed
* `asf_tools.water_map` now produces water extent rasters with pixels values of `1` where water is present and `0` where water is not present
* `asf_tools.water_map` now uses the  updated [ASF Global HAND dataset](https://glo-hand-30m.s3.amazonaws.com/readme.html) derived from the [2021 release of the Copernicus GLO-30 Public DEM](https://spacedata.copernicus.eu/blogs/-/blogs/copernicus-dem-2021-release-now-available)

## [0.4.3]

### Changed
* `asf_tools.water_map` now uses the  updated [ASF Global HAND dataset](https://copernicus-hand-30m.s3.amazonaws.com/) derived from the [2021 release of the Copernicus GLO-30 Public DEM](https://spacedata.copernicus.eu/blogs/-/blogs/copernicus-dem-2021-release-now-available)

## [0.4.2](https://github.com/ASFHyP3/asf-tools/compare/v0.4.1...v0.4.2)

### Added
* The accumulation threshold can now be specified in all HAND calculation functions and entry points:
  * like `calculate_hand`, `calculate_hand_for_basins` and `make_copernicus_hand` functions in 
    `asf_tools.hand.calculate` now accept an `acc_thresh` keyword argument
  * The `calculate_hand` console script entrypoint now accepts an `-a`/`--acc-threshold` argument
* `asf_tools.flood_map` now creates a cloud-optimized GeoTIFF of the perennial water mask used

### Fixed
* `asf_tools.flood_map` now correctly removes perennial water from the flood depth GeoTIFFs
* `asf_tools.flood_map` no longer calculates water/flood depth outside of the  RTC VV acquisition footprint

## [0.4.1](https://github.com/ASFHyP3/asf-tools/compare/v0.4.0...v0.4.1)

### Changed
* `asf_tools.dem` now uses the updated [2021 release of the Copernicus GLO-30 Public DEM](https://spacedata.copernicus.eu/blogs/-/blogs/copernicus-dem-2021-release-now-available)

### Fixed
* `asf_tools.hand.calculate` correctly uses [pyshed's `sGrid`](https://github.com/mdbartos/pysheds) for calculating HAND across all hydrobasins simultaneously. 
* `asf_tools.hand.calculate` will fill NaNs within the hydrobasins in the calculated HAND array with values interpolated from their neighbor's HOND (height of nearest drainage)
* `asf_tools.flood_map.iteartive` now produces more precise flood depth estimates by averaging water levels from a range of different initial guesses

## [0.4.0](https://github.com/ASFHyP3/asf-tools/compare/v0.3.3...v0.4.0)

### Added
* `asf_tools.flood_map` and an associated `flood_map` entrypoint for making
  flood depth maps with products generated by `asf_tools.water_map`. This
  functionality is still under active development and the products created
  using this function are likely to change in the future.

## [0.3.3](https://github.com/ASFHyP3/asf-tools/compare/v0.3.2...v0.3.3)

### Changed
* Upgraded `asf_tools`'s `pysheds` dependency to versions `>=0.3`

### Fixed
* `asf_tools.hand.calculate_hand` now explicitly uses `Pysheds.prid.Grid` because
  `sGrid` has no `add_gridded_data` attribute
* `calculate_hand` entrypoint now allows GDAL virtual file system (`/vsi*`) paths
  for the `hand_raster` and `vector_file` arguments

## [0.3.2](https://github.com/ASFHyP3/asf-tools/compare/v0.3.1...v0.3.2)

### Fixed
* [#99](https://github.com/ASFHyP3/asf-tools/issues/99) with better masked array handling

## [0.3.1](https://github.com/ASFHyP3/asf-tools/compare/v0.3.0...v0.3.1)

### Added
* We now provide an ASF Tools docker image: `ghcr.io/asfhyp3/asf-tools`. For usage, see the `asf_tools` [README](src/asf_tools/README.md).

### Changed
* `asf_tools.water_map` will raise a `ValueError` error if the HAND data is all zero

## [0.3.0](https://github.com/ASFHyP3/asf-tools/compare/v0.2.0...v0.3.0)

### Added
* `asf_tools.water_map` and an associated `water_map` entrypoint for making
  water extent maps using a multi-mode Expectation Maximization approach and refined using Fuzzy Logic
* `asf_tools.hand` sub package containing:
  * `asf_tools.hand.calculate` and an associated `calculate_hand` entrypoint for calculating Height Above 
    Nearest Drainage (HAND) from the Copernicus GLO-30 DEM
  * `asf_tools.hand.prepare` to prepare a raster from the Copernicus GLO-30 DEM derived Global HAND tiles
* `asf_tools.dem` to prepare a virtual raster (VRT) mosaic of the Copernicus GLO-30 DEM tiles
* `expectation_maximization_threshold` in `asf_tools.threshold` to calculate water threshold value
  using a multi-mode Expectation Maximization approach
* `tile_array` and `untile_array` in `asf_tools.tile` to transform a numpy array into a set of tiles
* `convert_scale` in `asf_tools.raster` to transform calibrated raster between decibel, power, and amplitude scales

### Changed
* ASF_Tools ArcGIS toolbox now accepts inputs in dB (decibel) scale for the RGBDecomp and ScaleConversion tools. 

## [0.2.0](https://github.com/ASFHyP3/asf-tools/compare/v0.1.1...v0.2.0)

### Added
* The `asf_tools` python package for working with Synthetic Aperture Radar (SAR) data. 
  See the [README](src/asf_tools/README.md)
* `asf_tools.composite` and an associated `make_composite` entrypoint for making
  mosaics using local resolution weighting (à la [David Smalls, 2012](https://doi.org/10.1109/IGARSS.2012.6350465))

### Changed
* This repository moved from `ASFHyP3/GIS-tools` to [`ASFHyP3/asf-tools`](https://github.com/ASFHyP3/asf-tools) 
  due to the broadening scope of the tools contained here

## [0.1.1](https://github.com/ASFHyP3/asf-tools/compare/v0.1.0...v0.1.1)

### Added
* Information and links to [On Demand RTC Processing](https://asfhyp3.github.io/using/vertex/) 
  in [Data Search - Vertex](https://search.asf.alaska.edu/) added to documentation

## [0.1.0](https://github.com/ASFHyP3/asf-tools/compare/v0.0.0...v0.1.0)

### Added
* RGB Decomposition tool to generate a color image from dual-pol SAR data, which 
  facilitates visual interpretation by decomposing the signals into surface 
  scattering with some volume scattering (red band), volume scattering (green band), 
  and surface scattering with very low volume scattering (blue band)
* Option to add raster outputs to the map automatically when tool processing is complete 
  (set as default for all tools producing raster outputs, but can be turned off in the 
  tool dialog if desired)
* `README.md` for the ArcGIS Toolbox to explain installation and usage
  
### Changed
* For the Scale Conversion and RGB Decomposition tools, the scale (amplitude/power) 
  input parameters are automatically populated when using input products that follow HyP3 
  naming scheme
