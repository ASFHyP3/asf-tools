# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/) 
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
