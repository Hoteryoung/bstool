from shapely.geometry import Polygon, MultiPolygon
import geojson


def polygon2mask(polygon):
    """convet polygon to mask

    Arguments:
        polygon {Polygon} -- input polygon (single polygon)

    Returns:
        list -- converted mask ([x1, y1, x2, y2, ...])
    """
    geo = geojson.Feature(geometry=polygon, properties={})
    if geo.geometry == None:
        return []
    coordinate = geo.geometry["coordinates"][0]     # drop the polygon of hole
    mask = []
    for idx, point in enumerate(coordinate):
        if idx == len(coordinate) - 1:
            break
        x, y = point
        mask.append(int(abs(x)))
        mask.append(int(abs(y)))
    return mask