from pathlib import Path
from typing import Iterator, List, Union

from osgeo import ogr

ogr.UseExceptions()


def get_features(vector_path: Union[str, Path]) -> List[ogr.Feature]:
    ds = ogr.Open(str(vector_path))
    layer = ds.GetLayer()
    return [feature for feature in layer]


def get_property_values_for_intersecting_features(geometry: ogr.Geometry, features: Iterator) -> bool:
    for feature in features:
        if feature.GetGeometryRef().Intersects(geometry):
            return True


def intersecting_feature_properties(geometry: ogr.Geometry, features: Iterator,
                                    feature_property: str) -> List[str]:
    property_values = []
    for feature in features:
        if feature.GetGeometryRef().Intersects(geometry):
            property_values.append(feature.GetField(feature_property))
    return property_values
