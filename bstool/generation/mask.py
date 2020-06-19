from shapely.geometry import Polygon, MultiPolygon
from skimage import measure


def generate_polygon(mask_image, 
                     min_area=20):
    contours = measure.find_contours(mask_image, 0.5, positive_orientation='low')

    polygons = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        if contour.shape[0] < 3:
            continue
        
        poly = Polygon(contour)
        if poly.area < min_area:
            continue
        poly = poly.simplify(1.0, preserve_topology=False)
        if poly.geom_type == 'MultiPolygon':
            for poly_ in poly:
                if poly.area < min_area:
                    continue
                polygons.append(poly_)
        else:
            polygons.append(poly)

    return polygons