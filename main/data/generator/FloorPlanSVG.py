from logging import warning
import sys, os, io, re
from itertools import permutations

import numpy as np
import nanoid.generate

from bs4 import BeautifulSoup
from cairosvg import svg2png

from PIL import Image, ImageDraw
from skimage.morphology import medial_axis
from rasterio.features import shapes
import cv2

import torch
from torch_geometric.data import Data

import HouseConfig as Mapper
from ImageUtil import (
    remakeImage,
    openRgb,
    getFloorplanPolygon,
    qualityReduction,
    addNoise,
    modifyHue,
    similarityFeature,
    ShapeCentroid,
    ShapeMerge,
    ShapeArea,
    ShapeMergeMultiple,
    FindRelevantPoints,
    applyTransformation,
    transparentToWhite,
    StringTransformParse,
    StringPointsParse,
    StringPathParse,
    PolygonSimplificationDescriptor
)


__all__ = ['FloorPlanSVG']


namechar = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

image_dim = (416, 416)
fix_width = 1200

font_list = [0, 2, 3, 4]

class FloorPlanSVG:
    def __init__(self, svg, asset_dir, ID):
        if os.path.exists(os.path.dirname(svg)):
            with open(svg) as f:
                svgstr = f.read()

        self.structure = BeautifulSoup(svgstr, "xml")
        self.asset_directory = asset_dir

        self.background = None
        self.floors = []
        self.texts = []
        self.ID = ID

        self.__preprocessing()


    def __INFO__(self):
        for idx,floor in enumerate(self.floors):
            print(f"<-- FLOOR {idx} -->\n")
            
            print(f"    ROOMS:")
            for i,el in enumerate(floor['rooms']):
                print(f"        {el['name']} {el['id']}")
                for p in el['bound']:
                    print(f"            ", end="")
                    for q in p:
                        print("%6s" % ("{:.1f}".format(q)), end=" ")
                    print()
            print(f"    BOUNDARIES:")
            for i,el in enumerate(floor['boundaries']):
                print(f"        {el['name']} {el['id']}")
                for p in el['bound']:
                    print(f"            ", end="")
                    for q in p:
                        print("%6s" % ("{:.1f}".format(q)), end=" ")
                    print()
            print(f"    FURNITURES:")
            for i,el in enumerate(floor['furnitures']):
                print(f"        {el['name']} {el['id']}")
                for p in el['bound']:
                    print(f"            ", end="")
                    for q in p:
                        print("%6s" % ("{:.1f}".format(q)), end=" ")
                    print()


    def __preprocessing(self):
        for text_tag in self.structure.select('text'):
            if 'id' in text_tag.parent.attrs and text_tag.parent.attrs['id'] == 'NameLabel':
                coord = self.__getTextCoord(text_tag)
                self.texts.append({
                    'VisibleText': None,
                    'Coord': coord
                })
            text_tag.extract()
        for control in self.structure.select('.SelectionControls'):
            control.extract()

        self.__extractFloor()
        self.__extractFurniture()
        self.__floorPolygonCorrection()
        # self.__polygonSimplification()
        # self.__INFO__()
    

    def __getTextCoord(self, text_tag):
        matrix1 = np.identity(3, dtype=np.float32)
        matrix2 = np.identity(3, dtype=np.float32)

        if text_tag.parent.parent.has_attr('transform'):
            matrix1 = StringTransformParse(text_tag.parent.parent['transform'])
        if text_tag.parent.parent.parent.has_attr('transform'):
            matrix2 = StringTransformParse(text_tag.parent.parent.parent['transform'])

        coord = np.array([[0],[0],[1]], dtype=np.float32)
        coord = np.dot(matrix1, coord)
        coord = np.dot(matrix2, coord)
        return coord


    def __preprocessGraph(self, room, boundary):
        top, left = float('inf'), float('inf')
        bottom, right = 0, 0
        for s in (room + boundary):
            top = min(top, np.min(s['bound'][0,:]))
            left = min(left, np.min(s['bound'][1,:]))
            bottom = max(bottom, np.max(s['bound'][0,:]))
            right = max(right, np.max(s['bound'][1,:]))
        
        img = (np.zeros((int(bottom - top + 2), int(right - left + 2), 2)) + 255).astype(np.uint8)
        pt = np.expand_dims(np.array([top, left]), axis=1)

        door_index = 0
        for s in boundary:
            ID = s['id']
            if Mapper.getBoundaryName(ID) != 'Door':
                continue
            color = (255, door_index)
            door_index += 1
            
            cv2.fillPoly(img, pts=[np.round((s['bound'][:2] - pt).T[:,::-1]).astype(int)], color=color)
        for i,s in enumerate(room):
            ID = s['id']
            color = (ID, i)

            cv2.fillPoly(img, pts=[np.round((s['bound'][:2] - pt).T[:,::-1]).astype(int)], color=color)

        # cv2.imwrite(f'saved{self.ID}.png', np.stack((img[:,:,0], img[:,:,1], img[:,:,1]), axis=2))
        return img, door_index, (top, left)

    
    def __extractGraph(self, floor):
        roomCnt = len(floor['rooms'])
        graph = np.zeros((roomCnt, roomCnt), dtype=np.int8) - 1
        
        img, doorCnt, roomBound = self.__preprocessGraph(floor['rooms'], floor['boundaries'])
        skel = medial_axis(img[:,:,1] < 255)

        roomDoor = np.zeros((doorCnt, roomCnt), dtype=np.bool8)

        def noise_softmax(x):
            x = np.array(x)
            x = 8 * x + 6 * np.random.rand(*x.shape)
            return np.exp(x) / np.exp(x).sum()

        for trans in [(1, 0),(-1, 0),(0, -1),(0, 1)]:
            shifted = np.roll(img, trans, axis=(1,0))
            def conn(coord):
                a = img[coord[0],coord[1],:]
                b = shifted[coord[0],coord[1],:]
                c = np.zeros(2) + 255
                if (a == c).all() or (b == c).all():
                    return -1
                if b[0] < 255 and a[0] < 255:
                    graph[b[1], a[1]] = 0
                    graph[a[1], b[1]] = 0
                elif b[0] == 255 and a[0] < 255:
                    roomDoor[b[1], a[1]] = True
                elif a[0] == 255 and b[0] < 255:
                    roomDoor[a[1], b[1]] = True
                return -1

            indexes = np.argwhere(((((img == shifted).all(-1)) ^ 1) & skel))
            np.apply_along_axis(conn, 1, indexes)

        # koneksi pintu
        if doorCnt > 0:
            def rconn(rms):
                idx = np.argwhere(rms).squeeze()
                try:
                    graph[idx[0],idx[1]] = 1
                    graph[idx[1],idx[0]] = 1
                except:
                    idx = 0
                return -1
            np.apply_along_axis(rconn, 1, roomDoor)

        # convert adjacency matrix to adjacency list
        edge_index = []
        edge_attr = []
        door_onehot = [[1, 0], [0, 1]]
        for i in range(roomCnt):
            for j in range(roomCnt):
                if graph[i, j] >= 0:
                    edge_index.append([i, j])
                    edge_attr.append(noise_softmax(door_onehot[graph[i, j]]))

        # target kelas ruangan sama fitur text
        y = []
        text_x = np.random.random((roomCnt, Mapper.lenRoom())) * 5

        for i,room in enumerate(floor['rooms']):
            y.append(Mapper.oneHotRoomID(room['id']))

            minx = np.argmin(text_x[i,:])

            if np.random.random() < 0.7:
                text_x[i,minx] /= 2

            miny = text_x[i,minx]
            text_x[i,[room['id'], minx]] = text_x[i,[minx, room['id']]]

            text_x[i,:] = 2 * miny - text_x[i,:]

            # softmax
            tmp = np.exp(text_x[i,:])
            text_x[i,:] = tmp / tmp.sum()

        # agregasi kelas furniture
        x = np.zeros((roomCnt, Mapper.lenIconBoundary()))
        top, left = roomBound

        for furniture in floor['furnitures']:
            center = ShapeCentroid(furniture['bound'])
            x_id = int(center[0]) - int(top)
            y_id = int(center[1]) - int(left)
            if x_id >= img.shape[0] or y_id >= img.shape[1] or x_id < 0 or y_id < 0:
                continue
            room_type, room_id = img[x_id, y_id, :]
            label = noise_softmax(Mapper.oneHotIconID(furniture['id']))
            
            if room_type < 255:
                x[room_id,:] = np.maximum(x[room_id,:], label)
        
        # concat x dan text_x
        x = np.hstack((x, text_x))

        x = torch.tensor(np.array(x), dtype=torch.float)
        y = torch.tensor(np.array(y), dtype=torch.float)
        
        if edge_index == []:
            edge_index = None
        else:
            edge_index = torch.tensor(np.array(edge_index), dtype=torch.long).transpose(0, 1)
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return graph


    def __extractFloor(self):
        self.setStyle('mask')
        self.__houseFloorVisibility(True)

        floorplans = self.structure.select("g[class=Floor]")
        floorplan_polygons = getFloorplanPolygon(
            self.__getImage('crispEdges')
        )

        fp_len = len(floorplan_polygons)
        f_len = len(floorplans)

        if fp_len > f_len:
            fp_area = [ShapeArea(e) for e in floorplan_polygons]
            while len(fp_area) > f_len:
                minimum = float('inf')
                min_idx = 0
                for idx,area in enumerate(fp_area):
                    if area < minimum:
                        min_idx = idx
                        minimum = area
                del floorplan_polygons[min_idx]
                del fp_area[min_idx]

        for idx,floorPoly in enumerate(floorplan_polygons):
            self.floors.append({
                'id': floorplans[idx].g['id'],
                'polygon': floorPoly.astype(np.uint16),
                'furnitures': [],
                'boundaries': [],
                'rooms': []
            })
    
    
    def __mergeRoom(self, rooms):
        new_rooms = []
        n_rooms = len(rooms)

        for i in range(n_rooms):
            if 'skip' in rooms[i] and rooms[i]['skip']:
                continue
            for j in range(i + 1, n_rooms):
                if rooms[i]['id'] != rooms[j]['id']:
                    continue
                merged = ShapeMerge(rooms[i]['bound'], rooms[j]['bound'])
                if merged is not None:
                    rooms[j]['skip'] = True

                    rooms[i] = {
                        'id': rooms[i]['id'],
                        'name': rooms[i]['name'],
                        'alias': rooms[i]['alias'],
                        'bound': merged
                    }
            new_rooms.append(rooms[i])
        return new_rooms


    def __extractFurniture(self):
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.setStyle('boundaries', color)
        # floorplans = self.structure.select("g[class=Floor]")
        self.__houseFloorVisibility(False)

        for idx,floor in enumerate(self.floors):
            struct = self.structure.select_one(f"g[id={floor['id']}]")

            for e in struct.select("g"):
                if not e.has_attr('class'):
                    continue
                if "FixedFurniture " in e.attrs['class']:
                    icon_name = e.attrs['class'].split(' ')[1]
                    icon_name = Mapper.getIconName(icon_name)
                    icon_ID = Mapper.getIconID(icon_name)

                    if icon_name is None or icon_ID is None:
                        continue

                    bound = self.__getFurnitureBoundingBox(e)
                    if bound is None:
                        bound = self.__getFurnitureBoundingBox(e, {"class": "InnerPolygon"})
                    if bound is None:
                        continue

                    self.floors[idx]['furnitures'].append({
                        'id': icon_ID,
                        'name': icon_name,
                        'bound': bound
                    })
                if "Space " in e.attrs['class']:
                    alias_room_name = e.attrs['class'].split(' ')[1]
                    try:
                        room_name = Mapper.getRoomName(alias_room_name)
                        room_ID = Mapper.getRoomID(room_name)
                    except:
                        warning(f"Room {room_name} doesn't exist")
                        continue
                    poly = StringPointsParse(e.polygon.attrs["points"])
                    
                    self.floors[idx]['rooms'].append({
                        'id': room_ID,
                        'name': room_name,
                        'alias': alias_room_name,
                        'bound': poly
                    })
                if 'id' not in e.attrs:
                    continue
                if e.attrs['id'] == 'Stairs':
                    bounds = []
                    for child in e:
                        bound = StringPointsParse(child.polygon.attrs["points"])
                        bounds.append(bound)
                    if len(bounds) <= 0:
                        continue
                    bound = np.hstack(bounds)

                    top = int(np.min(bound[0,:]))
                    left = int(np.min(bound[1,:]))
                    bottom = int(np.max(bound[0,:]))
                    right = int(np.max(bound[1,:]))

                    bound = np.array([
                        [top, bottom, bottom, top],
                        [left, left, right, right],
                        [1, 1, 1, 1]
                    ])

                    icon_name = Mapper.getIconName("Stairs")
                    icon_ID = Mapper.getIconID(icon_name)

                    self.floors[idx]['furnitures'].append({
                        'id': icon_ID,
                        'name': icon_name,
                        'bound': bound
                    })
                # if e.attrs['id'] == 'Wall':
                #     icon_name = 'Wall'
                #     icon_ID = Mapper.getBoundaryID(icon_name)

                #     bound = self.__getFurnitureBoundingBox(e, {"id": "Threshold"})
                #     if bound is None:
                #         continue

                #     self.floors[idx]['boundaries'].append({
                #         'id': icon_ID,
                #         'name': icon_name,
                #         'bound': bound
                #     })
            self.floors[idx]['rooms'] = self.__mergeRoom(self.floors[idx]['rooms'])

            struct.parent['style'] = ''
            img = self.__getImage("crispEdges")
            img = np.array(Image.fromarray(img))[:,:,:3]
            # cv2.imwrite("saved.png", img)

            boundaries_array = []
            
            for i,c in enumerate(color):
                mask = (img == np.array(c)).all(-1)
                F = mask.astype(np.uint16)

                poly = (s for _, (s, _) in enumerate(shapes(F, mask=mask)))

                for l in poly:
                    pts = np.array(l['coordinates'][0], dtype=np.int32)
                    # pts = np.apply_along_axis(lambda a: [1, *a], 1, pts)[:,::-1].T
                    pts = pts[:,::-1].T
                    
                    boundaries_array.append({
                        'id': i,
                        'name': Mapper.getBoundaryName(i),
                        'bound': pts
                    })
            
            polygon_descriptor = PolygonSimplificationDescriptor([
                e['bound'] for e in self.floors[idx]['rooms'] + boundaries_array
            ])

            for i in range(len(self.floors[idx]['rooms'])):
                self.floors[idx]['rooms'][i]['bound'] = polygon_descriptor(self.floors[idx]['rooms'][i]['bound'])
            
            boundaries_array = sorted(boundaries_array, key=lambda e: abs(e['id'] - Mapper.getBoundaryID('Wall')))
            img = np.zeros_like(img)
            for i,polygon in enumerate(boundaries_array):
                pts = polygon_descriptor(polygon['bound'])
                object_color = color[boundaries_array[i]['id']]

                if len(pts[0]) == 0:
                    continue
                cv2.fillPoly(img, pts=[(pts.T[:,::-1]).astype(int)], color=object_color)
            for room in self.floors[idx]['rooms']:
                cv2.fillPoly(img, pts=[(room['bound'].T[:,1::-1]).astype(int)], color=(255,255,255))
            
            for i,c in enumerate(color):
                mask = (img == np.array(c)).all(-1)
                F = mask.astype(np.uint16)

                poly = (s for _, (s, _) in enumerate(shapes(F, mask=mask)))

                for l in poly:
                    pts = np.array(l['coordinates'][0], dtype=np.int32)
                    # pts = np.apply_along_axis(lambda a: [1, *a], 1, pts)[:,::-1].T
                    pts = pts[:,::-1].T
                    
                    self.floors[idx]['boundaries'].append({
                        'id': i,
                        'name': Mapper.getBoundaryName(i),
                        'bound': pts
                    })

            struct.parent['style'] = "display: none;"
        self.__houseFloorVisibility(True)

    
    def __floorPolygonCorrection(self):
        floor_center = []
        attributes_center = []
        
        for idx,floor in enumerate(self.floors):
            floor_center.append((ShapeCentroid(floor['polygon']), idx))
            
            attribute_center = np.array([0, 0], dtype=float)
            count = 0
            for el in floor['boundaries'] + floor['rooms'] + floor['furnitures']:
                attribute_center += ShapeCentroid(el['bound'])
                count += 1
            attribute_center /= count
            attributes_center.append((attribute_center, idx))
        
        min_cost = float('inf')
        best_perm = []
        
        for perm in permutations(floor_center):
            cost = 0
            for i in range(self.getFloorplanCount):
                first_pt = perm[i][0]
                second_pt = attributes_center[i][0]
                cost += np.sum((first_pt - second_pt) * (first_pt - second_pt))
            if min_cost > cost:
                best_perm = perm
                min_cost = cost
        
        corrected_polygon = []
        for idx,(_,pr) in enumerate(best_perm):
            corrected_polygon.append(self.floors[pr]['polygon'].copy())
        for idx,poly in enumerate(corrected_polygon):
            self.floors[idx]['polygon'] = poly


    def __getFurnitureBoundingBox(self, furniture, selector={"class": "BoundaryPolygon"}):
        polygon = furniture.find("g", selector)
        if polygon is None:
            return None

        bounds = []
        names = []

        for child in polygon.findChildren():
            if child.name == 'polygon':
                bound = StringPointsParse(child["points"])
            elif child.name == 'rect':
                x = float(child['x'])
                y = float(child['y'])
                w = float(child['width'])
                h = float(child['height'])

                bound = np.array([
                    [y, y, y + h, y + h],
                    [x, x + w, x + w, x],
                    [1, 1, 1, 1]
                ])
            elif child.name == 'circle':
                cx = float(child['cx'])
                cy = float(child['cy'])
                r = float(child['r'])

                bound = StringPathParse(f"M {cx} {cy} m {-r}, 0 a {r},{r} 0 1,0 {r * 2},0 a {r},{r} 0 1,0 {-r * 2},0")
            elif child.name == 'path':
                bound = StringPathParse(child["d"])
            else:
                continue

            if furniture.has_attr('transform'):
                bound = applyTransformation(bound,
                    StringTransformParse(furniture['transform'])
                )

            if furniture.parent.has_attr('transform'):
                bound = applyTransformation(bound,
                    StringTransformParse(furniture.parent['transform'])
                )
            
            if bound.shape[1] >= 3:
                names.append(child.name)
                bounds.append(bound)

        if len(bounds) <= 0:
            return None

        try:
            bound = ShapeMergeMultiple(bounds)
        except Exception as e:
            print(names)
            print(bounds)
            raise e

        return bound.astype(np.uint16)


    def __houseFloorVisibility(self, visible=True):
        floorplans = self.structure.select("g[class=Floor]")
        
        display = "display: none;"
        if visible: display = ""
        
        for floorplan in floorplans:
            floorplan['style'] = display


    def __getImage(self, shape_rendering='geometricPrecision'):
        self.structure.svg.attrs['shape-rendering'] = shape_rendering

        svg = str(self.structure)
        mem = io.BytesIO()
        svg2png(bytestring=svg, write_to=mem)

        self.structure.svg.attrs['shape-rendering'] = 'crispEdges'
        return np.array(Image.open(mem))

    
    def __putRoomText(self, img, transformation):
        font = font_list[np.random.randint(len(font_list))]
        font_scale = 0.5
        thickness = 1

        for floor in self.floors:
            text_arr = []

            for room in floor['rooms']:
                text = room['alias']
                room_bound = applyTransformation(room['bound'], transformation, dtype=np.int32)

                if text.lower() == 'undefined':
                    continue

                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

                try:
                    point, area = FindRelevantPoints(room_bound, text_size)
                except Exception as e:
                    continue
                if point is None:
                    continue
                
                point = (point + np.array([text_size[1]/2, -text_size[0]/2])).astype(np.int32)

                text_arr.append((text, point, area))

            N = len(floor['rooms'])
            pick = int((np.random.rand() * 0.2 + 0.5) * N)
            text_arr.sort(key=lambda x: x[2], reverse=True)
            
            for text,point,_ in text_arr[:pick]:
                img = cv2.putText(img, text, point.squeeze()[1::-1], font, font_scale, (0,0,0), thickness, cv2.LINE_AA)
        
        return img


    def __createImage(self, homography=False):
        img = self.__getImage()
        
        transformation = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        if self.background is not None and not homography:
            img, transformation = remakeImage(
                img,
                self.background,
                scale=(1.3, 1.5),
                shift=(0.1, 0),
                W=fix_width
            )
        else:
            img = transparentToWhite(img)

        if not homography:
            img = self.__putRoomText(img, transformation)
            img = qualityReduction(img)
            img = modifyHue(img)
            img = addNoise(img, 'poisson')

        return img, transformation


    def __saveImage(self, img, directory, resize_dim=image_dim, bg=None):
        if bg == None:
            resized = img
        elif bg == 'resize':
            resized = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
        elif bg == 'fill':
            resized = (np.ones((*image_dim, 3)) * 255).astype(np.uint8)
            size = img.shape
            if size[0] > size[1]:
                newdim = (
                    image_dim[0],
                    size[1] * 416 // size[0]
                )
                img = cv2.resize(img, newdim[::-1], interpolation=cv2.INTER_AREA)
            else:
                newdim = (
                    size[0] * 416 // size[1],
                    image_dim[1]
                )
                img = cv2.resize(img, newdim[::-1], interpolation=cv2.INTER_AREA)

            pt = (
                image_dim[0] // 2 - newdim[0] // 2,
                image_dim[1] // 2 - newdim[1] // 2,
            )
            resized[pt[0] : pt[0]+newdim[0], pt[1] : pt[1]+newdim[1], :] = img
        cv2.imwrite(directory, resized[:,:,::-1])


    def __savePoly(self, transformation, shape, directory):
        with open(directory, "w") as f:
            first = True
            for floor in self.floors:
                transformed = applyTransformation(floor['polygon'], transformation)
                transformed[0,:] = transformed[0,:] / shape[0]
                transformed[1,:] = transformed[1,:] / shape[1]
                transformed = transformed.T.flatten()

                newline = '\n'
                if first:
                    newline = ''
                strg = f"{newline}0 {' '.join(np.char.mod('%f', transformed))}"
                first = False

                f.write(strg)


    def __saveBoundingBox(self, transformation, polygons, floorBound, directory, pre_transform=None):
        shape = (floorBound[2] - floorBound[0], floorBound[3] - floorBound[1])
        # mx = max(shape[0], shape[1])
        with open(directory, "w") as f:
            first = True
            for furniture in polygons:
                ID = furniture['id']
                transformed = applyTransformation(furniture['bound'], transformation)

                transformed[0,:] = (transformed[0,:] - floorBound[0]) / shape[0]
                transformed[1,:] = (transformed[1,:] - floorBound[1]) / shape[1]
                transformed = transformed.T.flatten()

                newline = '\n'
                if first:
                    newline = ''
                strg = f"{newline}{ID} {' '.join(np.char.mod('%f', transformed))}"
                first = False

                f.write(strg)
    
    
    def __saveGraph(self, floor, directory):
        graph = self.__extractGraph(floor)
        torch.save(graph, directory)
        return


    def setBackground(self, background):
        if not isinstance(background, str):
            return

        background_directory = f'{self.asset_directory}\\background\\{background}.jpg'

        if os.path.exists(os.path.dirname(background_directory)):
            s = openRgb(background_directory)
            self.background = s


    def setStyle(self, style=None, color=None):
        if style is None:
            for style_tag in self.structure.select('style'):
                style_tag.extract()

            tag = self.structure.new_tag("style")
            tag.string = '''
                svg{
                    fill: red;
                    stroke: red;
                }
            '''
            self.structure.svg.g.insert_before(tag)
            return

        if not isinstance(style, str):
            return
        
        style_directory = f'{self.asset_directory}\\style\\{style}.css'
        
        if os.path.exists(os.path.dirname(style_directory)):
            s = ""
            
            with open(style_directory) as f:
                s = f.read()
            
            if color is not None:
                trans = {}
                for i,c in enumerate(color):
                    trans[f'%%COLOR[{i}]%%'] = '#%02x%02x%02x' % c

                rep = dict((re.escape(k), v) for k, v in trans.items())
                pattern = re.compile("|".join(rep.keys()))
                s = pattern.sub(lambda m: rep[re.escape(m.group(0))], s)
        
            for style_tag in self.structure.select('style'):
                style_tag.extract()

            tag = self.structure.new_tag("style")
            tag.string = s
            self.structure.svg.g.insert_before(tag)


    @property
    def getFloorplanCount(self):
        return len(self.floors)


    def saveStructure(self, target_directory):
        img, transformation = self.__createImage()
        tmp_img = Image.fromarray(img.astype(np.uint8))
        
        roi_image_dir = f'{target_directory}/roi-detection/image/G{self.ID}.png'
        roi_poly_dir = f'{target_directory}/roi-detection/label/G{self.ID}.txt'
        self.__saveImage(img, roi_image_dir)
        self.__savePoly(transformation, img.shape, roi_poly_dir)

        for floor_idx,floor in enumerate(self.floors):
            transformed = applyTransformation(floor['polygon'], transformation, dtype=np.int32)

            mask = Image.new('L', tmp_img.size, 0)
            ImageDraw.Draw(mask).polygon(transformed[::-1].T.flatten().tolist(), outline=255, fill=255)

            top = int(np.min(transformed[0,:]))
            left = int(np.min(transformed[1,:]))
            bottom = int(np.max(transformed[0,:]))
            right = int(np.max(transformed[1,:]))

            cropped = Image.new("RGB", tmp_img.size, (255, 255, 255))
            cropped.paste(tmp_img, mask=mask)
            cropped = np.asarray(cropped)[top:bottom,left:right]

            # save gambar
            fileIDsymbol = nanoid.generate(namechar, 24)
            symbol_image_dir = f'{target_directory}/symbol-detection/image/G{self.ID}F{floor_idx + 1}.png'
            symbol_poly_dir = f'{target_directory}/symbol-detection/label/G{self.ID}F{floor_idx + 1}.txt'

            self.__saveImage(cropped, symbol_image_dir)
            self.__saveBoundingBox(transformation, floor['furnitures'] + floor['boundaries'], [top, left, bottom, right], symbol_poly_dir)

            # save graph
            fileIDroom = nanoid.generate(namechar, 24)
            room_image_dir = f'{target_directory}/room-classification/graph/G{self.ID}F{floor_idx + 1}.pt'

            self.__saveGraph(floor, room_image_dir)


    def homographyChecking(self, transformation, shape):
        transformed = applyTransformation(self.floors[0]['polygon'], transformation, dtype=np.int32)
        transformed[0,:] = np.clip(transformed[0,:], a_min=0, a_max=shape[0])
        transformed[1,:] = np.clip(transformed[1,:], a_min=0, a_max=shape[1])
        
        area = ShapeArea(transformed)

        prop = area / (shape[0] * shape[1])
        
        return prop > 0.1

    
    def displayOriginalImage(self, image_directory, similarity_param):
        self.setStyle()

        sim_img, _ = self.__createImage(homography=True)
        img = openRgb(image_directory)
        shape = img.shape

        transformation = similarityFeature(sim_img, img, **similarity_param)
        if transformation is None:
            return None, None

        pts = []

        for floor_idx,floor in enumerate(self.floors):
            transformed = applyTransformation(floor['polygon'], transformation, dtype=np.int32)
            pts.append(np.array([np.clip(transformed[1], 0,shape[1] - 1), np.clip(transformed[0], 0,shape[0] - 1)]).T.reshape((-1, 1, 2)))

            for fur in floor['furnitures'] + floor['boundaries'] + floor['rooms']:
                transformed = applyTransformation(fur['bound'], transformation, dtype=np.int32)
                pts.append(np.array([np.clip(transformed[1], 0,shape[1] - 1), np.clip(transformed[0], 0,shape[0] - 1)]).T.reshape((-1, 1, 2)))
        img = cv2.polylines(img, pts, True, (255, 0, 0), 1)
        
        return img[:,:,::-1], transformation


    def saveOriginalImage(self, image_directory, target_directory, similarity_param, transformation=None):
        self.setStyle()

        sim_img, _ = self.__createImage(homography=True)
        img = openRgb(image_directory)

        if transformation is None:
            transformation = similarityFeature(sim_img, img, **similarity_param)
            if not self.homographyChecking(transformation, img.shape):
                return
        if transformation is None:
            return

        tmp_img = Image.fromarray(img.astype(np.uint8))

        roi_image_dir = f'{target_directory}/roi-detection/image/O{self.ID}.png'
        roi_poly_dir = f'{target_directory}/roi-detection/label/O{self.ID}.txt'
        self.__saveImage(img, roi_image_dir)
        self.__savePoly(transformation, img.shape, roi_poly_dir)
        
        for floor_idx,floor in enumerate(self.floors):
            transformed = applyTransformation(floor['polygon'], transformation, dtype=np.int32)

            mask = Image.new('L', tmp_img.size, 0)
            ImageDraw.Draw(mask).polygon(transformed[::-1].T.flatten().tolist(), outline=255, fill=255)

            top = int(max(np.min(transformed[0,:]), 0))
            left = int(max(np.min(transformed[1,:]), 0))
            bottom = int(min(np.max(transformed[0,:]), img.shape[0]))
            right = int(min(np.max(transformed[1,:]), img.shape[1]))

            cropped = Image.new("RGB", tmp_img.size, (255, 255, 255))
            cropped.paste(tmp_img, mask=mask)
            cropped = np.asarray(cropped)[top:bottom,left:right]

            # save gambar
            symbol_image_dir = f'{target_directory}/symbol-detection/image/O{self.ID}F{floor_idx + 1}.png'
            symbol_poly_dir = f'{target_directory}/symbol-detection/label/O{self.ID}F{floor_idx + 1}.txt'

            self.__saveImage(cropped, symbol_image_dir)
            self.__saveBoundingBox(transformation, floor['furnitures'] + floor['boundaries'], [top, left, bottom, right], symbol_poly_dir)