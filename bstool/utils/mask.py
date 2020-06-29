
def single_valid_polygon(polygon):
    if not polygon.is_valid:
        return False
    elif polygon.geom_type not in ['Polygon', 'MultiPolygon']:
        return False
    else:
        return True