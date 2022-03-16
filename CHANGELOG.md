# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/) 
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2](https://github.com/ASFHyP3/asf-tools/compare/v0.3.1...v0.3.2)

### Fixed
* [#99](https://github.com/ASFHyP3/asf-tools/issues/99) with better masked array handling

## [0.3.1](https://github.com/ASFHyP3/asf-tools/compare/v0.3.0...v0.3.1)

### Added
* We now provide an ASF Tools docker image: `ghcr.io/asfhyp3/asf-tools`. For usage, see the `asf_tools` [README](asf_tools/README.md).

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
  See the [README](asf_tools/README.md)
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
