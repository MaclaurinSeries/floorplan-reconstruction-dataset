from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np

__all__ = ['ShapeCentroid', 'ShapeMerge']


def ShapeCentroid(poly):
    p = Polygon(poly[:2].T)
    return np.array(p.centroid)


def ShapeMerge(poly1, poly2):
    p1 = Polygon(poly1[:2].T)
    p2 = Polygon(poly2[:2].T)

    merged = unary_union([p1, p2])
    if len(merged.geoms) > 1:
        return None
    
    return merged.geoms[0]