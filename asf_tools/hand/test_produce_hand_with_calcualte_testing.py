import argparse
import logging
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime

from osgeo import gdal
from shapely.geometry import GeometryCollection, shape

from asf_tools.hand.calculate_testing import calculate_hand_for_basins
from asf_tools.dem import prepare_dem_vrt
from asf_tools.hand.upload_to_s3 import upload_file


log = logging.getLogger(__name__)


def make_asf_hand(dem_tile: str) -> str:
    dem_info = gdal.Info(dem_tile, format='json')
    dem_geometry = shape(dem_info['wgs84Extent'])

    dem_buffered = GeometryCollection([dem_geometry.buffer(0.5)])

    with NamedTemporaryFile(suffix='.vrt', delete=False) as buffered_dem_vrt:
        prepare_dem_vrt(buffered_dem_vrt.name, dem_buffered)

    out_raster = Path(dem_tile).name.replace('DEM.tif', 'HAND.tif')
    calculate_hand_for_basins(out_raster, dem_buffered, buffered_dem_vrt.name)

    # TODO: Mask ocean pixels

    # TODO: crop buffered HAND back to original DEM tile size

    return out_raster


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('dem_tile', help='DEM GeoTIFF to calculate HAND from. Raster file Must be openable by GDAL,'
                                         ' see: https://gdal.org/drivers/vector/index.html')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))
    log.info(f'Making HAND from {args.dem_tile}')

    hand_raster = make_asf_hand(args.dem_tile)

    # for test purpose
    '''
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    hand_raster = f"test_batch_{date_time}"
    f = open(hand_raster,"w")
    message=f"if you see me at s3, it means the file {hand_raster} {args.dem_tile} is uploaded from container to s3 is successful"
    print(message)
    f.write(message)
    f.close()
    '''

    log.info(f'HAND GeoTIFF created successfully: {hand_raster}')

    # TODO: upload to S3? I'd like the new HAND we calculate to be laid out exactly like the COP DEM bucket. So, if you
    #       know the COP DEM S3 path/url all you'd have to do is replace the bucket name and `_DEM` with `_HAND`

    # upload the  {hand_raster} to s3

    # upload_file(hand_raster, "jzhu4", "global_hand")


if __name__ == '__main__':
    main()
