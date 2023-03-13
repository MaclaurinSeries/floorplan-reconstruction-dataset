from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import polygonize, unary_union, transform
import numpy as np

__all__ = ['ShapeCentroid', 'ShapeArea', 'ShapeMerge', 'ShapeMergeMultiple', 'applyTransformation', 'FindRelevantPoints', 'PolygonSimplificationDescriptor']


def ShapeCentroid(poly):
    if poly.shape[1] <= 2:
        return np.sum(poly, axis=1) / poly.shape[1]
    p = Polygon(poly[:2].T)
    return np.array(p.centroid)


def ShapeMerge(poly1, poly2):
    p1 = Polygon(poly1[:2].T)
    p2 = Polygon(poly2[:2].T)

    merged = unary_union([p1, p2])
    if isinstance(merged, MultiPolygon):
        if len(merged.geoms) > 1:
            return None
        
        x,y = merged.geoms[0].exterior.coords.xy
    else:
        x,y = merged.exterior.coords.xy

    ones = np.ones(len(x))

    return np.vstack((x, y, ones))


def ShapeMergeMultiple(poly):
    merged = unary_union([Polygon(p[:2].T) for p in poly])
    if isinstance(merged, MultiPolygon):
        if len(merged.geoms) > 1:
            merged_area = np.array([geom.area for geom in merged.geoms])
            choosen = merged.geoms[np.argmax(merged_area)]
        else:
            choosen = merged.geoms[0]
        x,y = choosen.exterior.coords.xy
    else:
        x,y = merged.exterior.coords.xy

    ones = np.ones(len(x))

    return np.vstack((x, y, ones))


def ShapeArea(poly):
    p = Polygon(poly[:2].T)
    if not p.is_valid:
        p = p.buffer(0)
    return p.area


def applyTransformation(poly, transformation, dtype=np.float32):
    poly = np.array(poly)
    if poly.shape[0] < 3:
        value = np.vstack((poly, np.ones((poly.shape[1]))))
    else:
        value = poly[:3,:]
    return np.dot(transformation, value)[:2].astype(dtype)


def FindRelevantPoints(poly, size):
    poly = ShrinkPolygon(
        np.array(poly),
        size
    )

    centroids = []
    for p in poly:
        if p.is_empty:
            continue
        centroids.append({
            'area': p.area,
            'coord': np.array(p.centroid)
        })
    
    if len(centroids) == 0:
        return None, None

    centroids.sort(key=lambda c: c['area'], reverse=True)

    return centroids[0]['coord'], centroids[0]['area']


def ShrinkPolygon(poly, size):
    w, h = size
    c = w / h
    poly[0,:] = poly[0,:] * c
    
    p = Polygon(poly[:2].T)
    p_shrink = p.buffer(-w/2, cap_style=2, join_style=2)

    coords = []

    if isinstance(p_shrink, MultiPolygon):
        for p in p_shrink.geoms:
            coords.append(transform(lambda x,y,z=None: (x/c, y), p))
    else:
        coords.append(transform(lambda x,y,z=None: (x/c, y), p_shrink))

    return coords


def SplitSelfIntersectingPolygon(poly):
    poly = np.array(poly)
    ls = LineString(poly)
    lr = LineString(ls.coords[:] + ls.coords[0:1])

    mls = unary_union(lr)
    return list(polygonize(mls))


def arraySymplificationDescriptor(array, min_delta=2):
    array, count = np.unique(np.sort(array), return_counts=True)
    difference = array - np.roll(array, 1)
    x_sorted_diff = np.argsort(difference)

    mean = array.copy()

    # disjoint set
    parent = np.arange(array.shape[0])
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        i = find(i)
        j = find(j)
        if i == j:
            return
        parent[i] = j
        mean[j] = (mean[i] * count[i] + mean[j] * count[j]) / (count[i] + count[j])
        count[j] = count[i] + count[j]
    def mean_value(i):
        i = find(i)
        return mean[i]
    
    for arg in x_sorted_diff:
        if arg <= 0: continue
        if difference[arg] > min_delta: break
    
        if abs(mean_value(arg) - mean_value(arg - 1)) <= min_delta:
            union(arg, arg - 1)

    for i in range(array.shape[0]):
        parent[i] = find(i)
    
    element = array[np.unique(parent)]

    def descriptor(x):
        arg = np.argmin(np.abs(element - x))
        return element[arg]
    
    return descriptor


def PolygonSimplificationDescriptor(polygon_array):
    polygon_array = [pts[:2] for pts in polygon_array]

    point_set = np.hstack(polygon_array)
    y_set = point_set[0,:]
    x_set = point_set[1,:]

    y_desc = arraySymplificationDescriptor(y_set)
    x_desc = arraySymplificationDescriptor(x_set)

    def descriptor(polygon):
        npoly = np.apply_along_axis(lambda x: [y_desc(x[0]), x_desc(x[1])], 0, polygon)
        npoly_roll = np.roll(npoly, shift=1, axis=1)
        # npoly_roll[:,0] = [-1, -1]
        mask = (npoly == npoly_roll).all(axis=0)

        return npoly[:,~mask]

    return descriptor