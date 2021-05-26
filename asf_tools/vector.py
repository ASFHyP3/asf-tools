from pathlib import Path
from typing import Iterator, List, Union

from osgeo import ogr

ogr.UseExceptions()


def get_features(vector_path: Union[str, Path]) -> List[ogr.Feature]:
    ds = ogr.Open(str(vector_path))
    layer = ds.GetLayer()
    return [feature for feature in layer]


def intersects_features(geometry: ogr.Geometry, features: Iterator) -> bool:
    for feature in features:
        if feature.GetGeometryRef().Intersects(geometry):
            return True


def intersecting_feature_properties(geometry: ogr.Geometry, features: Iterator,
                                    feature_property: str) -> List[str]:
    file_paths = []
    for feature in features:
        if feature.GetGeometryRef().Intersects(geometry):
            file_paths.append(feature.GetField(feature_property))
    return file_paths
