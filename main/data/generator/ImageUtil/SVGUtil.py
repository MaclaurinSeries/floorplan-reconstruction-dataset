import numpy as np
from svgpath2mpl import parse_path


__all__ = ['StringTransformParse', 'StringPointsParse', 'StringPathParse']

def StringTransformParse(str_path):
    matrix = str_path[str_path.find("(") + 1 : str_path.find(")")]
    matt = list(map(np.float32, matrix.split(",")))
    matrix = np.array([
        [matt[3], matt[1], matt[5]],
        [matt[2], matt[0], matt[4]],
        [.0, .0, 1.0]
    ])

    return matrix


def StringPointsParse(str_path):
    poly = str_path.split(' ')
    points = []
    for p in poly:
        try:
            points.append([1, *list(map(np.float32, p.split(',')[:2]))])
        except:
            continue
    poly = np.array(points)
    
    return poly.T[::-1,:]


def StringPathParse(str_path):
    '''
    https://github.com/nvictus/svgpath2mpl
    '''
    path = parse_path(str_path)
    coords = path.to_polygons()

    return coords[0][:,::-1].T