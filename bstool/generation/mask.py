from shapely.geometry import Polygon, MultiPolygon
from skimage import measure
import bstool


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
                if poly_.area < min_area:
                    continue
                valid_flag = bstool.single_valid_polygon(poly_)
                if not valid_flag:
                    continue
                polygons.append(poly_)
        elif poly.geom_type == 'Polygon':
            valid_flag = bstool.single_valid_polygon(poly)
            if not valid_flag:
                continue
            polygons.append(poly)
        else:
            continue

    return polygons