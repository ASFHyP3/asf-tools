import pytest

from asf_tools import composite


def test_get_epsg_code():
    info = {'coordinateSystem': {'wkt': 'PROJCS["WGS 84 / UTM zone 54N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",141],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32654"]]'}}
    assert composite.get_epsg_code(info) == 32654

    info = {'coordinateSystem': {'wkt': 'PROJCS["WGS 84 / UTM zone 22N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-51],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32622"]]'}}
    assert composite.get_epsg_code(info) == 32622

    info = {'coordinateSystem': {'wkt': 'PROJCS["WGS 84 / UTM zone 33S",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32733"]]'}}
    assert composite.get_epsg_code(info) == 32733

    info = {'coordinateSystem': {'wkt': 'PROJCS["NAD83 / Alaska Albers",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",55],PARAMETER["standard_parallel_2",65],PARAMETER["latitude_of_center",50],PARAMETER["longitude_of_center",-154],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3338"]]'}}
    assert composite.get_epsg_code(info) == 3338


def test_get_target_epsg_code():
    # Northern hemisphere
    assert composite.get_target_epsg_code([32601]) == 32601
    assert composite.get_target_epsg_code([32601, 32601]) == 32601

    # both hemispheres
    assert composite.get_target_epsg_code([32601, 32702]) == 32601
    assert composite.get_target_epsg_code([32702, 32601]) == 32701

    # Southern hemisphere
    assert composite.get_target_epsg_code([32760]) == 32760
    assert composite.get_target_epsg_code([32730, 32732]) == 32731

    # antimeridian
    assert composite.get_target_epsg_code([32701, 32760]) == 32760
    assert composite.get_target_epsg_code([32701, 32760, 32701]) == 32701
    assert composite.get_target_epsg_code([32701, 32760, 32760]) == 32760

    assert composite.get_target_epsg_code(
        [32731, 32631, 32731, 32631, 32732, 32633, 32733, 32633, 32733]
    ) == 32732

    # bounds
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32600])
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32661])
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32700])
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32761])
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32601, 99, 32760])


def test_get_area_raster():
    raster = 'S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_VV.tif'
    assert composite.get_area_raster(raster) == 'S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_area.tif'

    raster = './foo/S1B_IW_20181104T030247_DVP_RTC30_G_gpuned_9F91_VH.tif'
    assert composite.get_area_raster(raster) == './foo/S1B_IW_20181104T030247_DVP_RTC30_G_gpuned_9F91_area.tif'

    raster = '/tmp/bar/S1B_IW_20181102T031956_DVP_RTC30_G_gpuned_1259_HH.tif'
    assert composite.get_area_raster(raster) == '/tmp/bar/S1B_IW_20181102T031956_DVP_RTC30_G_gpuned_1259_area.tif'
