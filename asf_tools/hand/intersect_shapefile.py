import argparse
import logging
import warnings
import numpy as np
import rasterio
import rasterio.mask
import fiona
import shapely
from tqdm.auto import tqdm
from shapely.geometry import GeometryCollection, shape, mapping


def fiona_read_vectorfile(vectorfile, get_property=None):
    """shapes=fiona_read_vectorfile(vectorfile, get_property=None)
       shapes, props=fiona_read_vectorfile(vectorfile, get_property='Property_Name')
       Returns a list of shapes (and optionally properties) using fiona.

       vectorfile: any fiona compatible vector file.
       get_property: String for the property to be read.
       shapes: List of vector "geometry"
       props:  List of vector "properties"
    """
    with fiona.open(vectorfile, "r") as shpf:
        shapes = [feature["geometry"] for feature in shpf]
        print(f"Number of shapes loaded: {len(shapes)}")
        if get_property is not None:
            props = [feature["properties"][get_property] for feature in shpf ]
            return shapes, props
        else:
            return shapes


def intersect_orig(shapes, polygon, properties=None):
    """
    polygons=intersect(shapes, polygon, properties=None)
    Returns polygons from multiple 'geometries' read by fiona.

    shapes: shapes returned by fiona_read_vectorfile()
    polygon: a single polygon to intersect with shapes
    properties: If not none, returns the property value instead of polygon geometry.
    """
    # first loop to split multi polygons to single polygons
    polygons = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, shape in enumerate(tqdm(shapes)):
            if shape['type'] == 'MultiPolygon':
                for l, p in enumerate(shape['coordinates']):
                    s = shapely.geometry.Polygon(p[0])
                    if polygon.intersects(s) and properties is None:
                        polygons.append(s)
                    elif polygon.intersects(s) and properties is not None:
                        if np.isscalar(properties[k]):
                            polygons.append(properties[k])
                        else:
                            polygons.append(properties[k][l])

            elif shape['type'] == 'Polygon':
                s = shapely.geometry.Polygon(shape['coordinates'][0])
                if polygon.intersects(s) and properties is None:
                    polygons.append(s)
                elif polygon.intersects(s) and properties is not None:
                    polygons.append(properties[k])
    return polygons


def get_bounding_box(demfile):
    with rasterio.open(demfile,'r') as dem:
        bounds = dem.bounds
        bounding_box = (bounds[0],bounds[1]), (bounds[2],bounds[1]), (bounds[2],bounds[3]), (bounds[0],bounds[3])
        return bounding_box


def intersect(shpfile, polygon, out_shpfile):

    with fiona.open(shpfile, 'r') as shp:
        schema = shp.schema
        # creation of the new shapefile with the intersection
        with fiona.open(out_shpfile, 'w', driver='ESRI Shapefile', schema=schema) as output:
            for basin in shp:
                if shape(basin['geometry']).intersects(polygon):
                    output.write(basin)
        return out_shpfile

def get_shapefile(in_shpfile, demfile, out_shpfile):

    # get the bounding box
    bbox = get_bounding_box(demfile)
    bbox_poly = []
    bbox_poly.extend(bbox)
    bbox_poly.append(bbox[0])
    dem_poly = shapely.geometry.Polygon(bbox_poly)

    outfile = intersect(in_shpfile, dem_poly, out_shpfile)

    return out_shpfile


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('shpfile', help='input hybas shapefile')
    parser.add_argument('demfile', help='input dem geotiff')
    parser.add_argument('outfile', help='output hybas shapefile')

    args = parser.parse_args()

    # shpfile="/media/jzhu4/data/hand/hybas_data/hybas_as_lev12_v1c.shp"

    # demfile = "/media/jzhu4/data/hand/Copernicus_DSM_COG_10_N23_00_E090_00_DEM.tif"

    # outfile = "/media/jzhu4/data/hand/Copernicus_DSM_COG_10_N23_00_E090_00_DEM_test.shp"

    # get the bounding box

    out_shpfile = get_shapefile(args.shpfile, args.demfile, args.outfile)

    print("complete...")


