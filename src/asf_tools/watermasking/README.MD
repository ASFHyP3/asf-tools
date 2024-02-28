These scripts are for creating a global (or regional) water mask dataset based off of a OpenStreetMaps, and optionally augmented by ESA WorldCover.

The basic steps are as follows:
    Downloads:
        1. Download the "Latest Weekly Planet PBF File" file from here: "https://planet.openstreetmap.org/".
        2. Download the WGS84 water polygons shapefile from: "https://osmdata.openstreetmap.de/data/water-polygons.html".
        3. The files should be unzipped and you should have something like `planet.osm.pbf` or `planet.pbf` and `water_polygons.shp` (and the support files for `water_polygons.shp`). 
    Scripts:
        1. Run ```generate_osm_dataset --planet-file-path [path-to-planet.pbf] --ocean-polygons-path [path-to-water-polygons.shp] --lat-begin -85 --lat-end 85 --lon-begin -180 --lon-end 180 --tile-width 5 --tile-height 5```
        2. Run ```fill_missing_tiles --fill-value 0 --lat-begin -90 --lat-end -85 --lon-begin -180 --lon-end 180 --tile-width 5 --tile-height 5```
        3. Run ```fill_missing_tiles --fill-value 1 --lat-begin 85 --lat-end 90 --lon-begin -180 --lon-end 180 --tile-width 5 --tile-height 5```