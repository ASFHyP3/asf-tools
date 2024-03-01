These scripts are for creating a global (or regional) water mask dataset based off of OpenStreetMaps, and optionally augmented by ESA WorldCover.

For the OSM water mask dataset, follow these steps to replicate our dataset:

1. Install osmium-tool from conda-forge or "https://osmcode.org/osmium-tool/".
2. Download the "Latest Weekly Planet PBF File" file from here: "https://planet.openstreetmap.org/".
3. Download the WGS84 water polygons shapefile from: "https://osmdata.openstreetmap.de/data/water-polygons.html".
4. The files should be unzipped and you should have something like `planet.osm.pbf` or `planet.pbf` and `water_polygons.shp` (and the support files for `water_polygons.shp`). 
5. Run ```generate_osm_dataset --planet-file-path [path-to-planet.pbf] --ocean-polygons-path [path-to-water-polygons.shp] --lat-begin -85 --lat-end 85 --lon-begin -180 --lon-end 180 --tile-width 5 --tile-height 5```
6. Run ```fill_missing_tiles --fill-value 0 --lat-begin -90 --lat-end -85 --lon-begin -180 --lon-end 180 --tile-width 5 --tile-height 5```
7. Run ```fill_missing_tiles --fill-value 1 --lat-begin 85 --lat-end 90 --lon-begin -180 --lon-end 180 --tile-width 5 --tile-height 5```

For the WorldCover water mask dataset, follow these steps:

1. Download the portions of the dataset for the areas you would like to cover from here: "https://worldcover2020.esa.int/downloader"
2. Extract the contents into a folder. Note, if you download multiple portions of the dataset, extract them all into the same folder.
3. Run ```generate_worldcover_dataset --worldcover-tiles-dir [path-to-worldcover-data] --lat-begin 55 --lat-end 80 --lon-begin -180 --lon-end 180 --tile-width 5 --tile-height 5```

Note that we only use WorldCover data over Alaska, Canada, and Russia for our dataset.
