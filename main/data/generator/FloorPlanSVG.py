from logging import warning
import sys, os, io, re

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
    ShapeCentroid,
    ShapeMerge
)


__all__ = ['FloorPlanSVG']


namechar = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

image_dim = (416, 416)
fix_width = 1200

class FloorPlanSVG:
    def __init__(self, svg, asset_dir):
        s = svg

        if os.path.exists(os.path.dirname(svg)):
            with open(svg) as f:
                svgstr = f.read()

        self.structure = BeautifulSoup(svgstr, "xml")
        self.asset_directory = asset_dir

        self.background = None
        self.floors = []
        self.texts = []

        self.__preprocessing()


    def __preprocessing(self):
        for text_tag in self.structure.select('text'):
            if 'id' in text_tag.parent.attrs and text_tag.parent.attrs['id'] == 'NameLabel':
                coord = self.__getTextCoord(text_tag)
                self.texts.append({
                    'VisibleText': None,
                    'Coord': coord
                })
            text_tag.extract()

        self.__extractFloor()
        self.__extractFurniture()
    

    def __getTextCoord(self, text_tag):
        matrix1 = np.identity(3, dtype=np.float32)
        matrix2 = np.identity(3, dtype=np.float32)

        if text_tag.parent.parent.has_attr('transform'):
            matrix1 = text_tag.parent.parent['transform']
            matrix1 = matrix1[matrix1.find("(") + 1 : matrix1.find(")")]
            matt = list(map(np.float32, matrix1.split(",")))
            matrix1 = np.array([
                [matt[3], matt[1], matt[5]],
                [matt[2], matt[0], matt[4]],
                [.0, .0, 1.0]
            ])
        if text_tag.parent.parent.parent.has_attr('transform'):
            matrix2 = text_tag.parent.parent.parent['transform']
            matrix2 = matrix2[matrix2.find("(") + 1 : matrix2.find(")")]
            matt = list(map(np.float32, matrix2.split(",")))
            matrix2 = np.array([
                [matt[3], matt[1], matt[5]],
                [matt[2], matt[0], matt[4]],
                [.0, .0, 1.0]
            ])

        coord = np.array([[0],[0],[1]], dtype=np.float32)
        coord = np.dot(matrix1, coord)
        coord = np.dot(matrix2, coord)
        return coord


    def __preprocessGraph(self, room, boundary):
        top, left = 10000, 10000
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
        return img, door_index, (top, left)

    
    def __extractGraph(self, floor):
        roomCnt = len(floor['rooms'])
        graph = np.zeros((roomCnt, roomCnt), dtype=np.int8) - 1
        
        img, doorCnt, roomBound = self.__preprocessGraph(floor['rooms'], floor['boundaries'])
        skel = medial_axis(img[:,:,1] < 255)

        roomDoor = np.zeros((doorCnt, roomCnt), dtype=np.bool8)

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

        def rconn(rms):
            idx = np.argwhere(rms).squeeze()
            graph[idx[0],idx[1]] = 1
            graph[idx[1],idx[0]] = 1
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
                    edge_attr.append(door_onehot[graph[i, j]])

        # target kelas ruangan
        y = []
        for room in floor['rooms']:
            y.append(Mapper.oneHotRoomID(room['id']))
        
        # agregasi kelas furniture
        x = np.zeros((roomCnt, Mapper.lenIconBoundary()))
        top, left = roomBound

        for furniture in floor['furnitures']:
            center = ShapeCentroid(furniture['bound'])
            room_type, room_id = img[int(center[0]) - top, int(center[1]) - left, :]
            label = Mapper.oneHotIconID(furniture['id'])
            
            if room_type < 255:
                x[room_id,:] = np.maximum(x[room_id,:], label)

        def softmax(x):
            return np.exp(x) / np.exp(x).sum()
        def noise(x):
            return 8 * x + np.random.rand(*x.shape)

        x = torch.tensor(np.apply_along_axis(softmax, 1, noise(x)), dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        edge_index = torch.tensor(np.apply_along_axis(softmax, 1, noise(edge_index)), dtype=torch.long).transpose(0, 1)
        edge_attr = torch.tensor(np.apply_along_axis(softmax, 1, noise(edge_attr)), dtype=torch.float)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return graph


    def __extractFloor(self):
        self.setStyle('mask')
        self.__houseFloorVisibility(True)

        floorplans = self.structure.select("g[class=Floor]")

        for idx,floorPoly in enumerate(getFloorplanPolygon(
            self.__getImage('crispEdges')
        )):
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

                    # TODO: blm selesai
        return rooms


    def __extractFurniture(self):
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.setStyle('boundaries', color)
        floorplans = self.structure.select("g[class=Floor]")
        self.__houseFloorVisibility(False)

        for idx,floor in enumerate(self.floors):
            struct = self.structure.find("g", {"id": floor['id']})

            for e in struct.select("g"):
                
                if "FixedFurniture " in e.attrs['class']:
                    icon_name = e.attrs['class'].split(' ')[1]
                    icon_name = Mapper.getIconName(icon_name)
                    icon_ID = Mapper.getIconID(icon_name)

                    if icon_name is None or icon_ID is None:
                        continue

                    bound = self.__getFurnitureBoundingBox(e)

                    self.floors[idx]['furnitures'].append({
                        'id': icon_ID,
                        'name': icon_name,
                        'bound': bound
                    })
                if "Space " in e.attrs['class']:
                    room_name = e.attrs['class'].split(' ')[1]
                    try:
                        room_name = Mapper.getRoomName(room_name)
                        room_ID = Mapper.getRoomID(room_name)
                    except:
                        warning(f"Room {room_name} doesn't exist")
                        continue
                    poly = e.polygon.attrs["points"]
                    poly = poly.split(' ')[:-1]
                    poly = np.array([[1, *list(map(np.float32, a.split(',')))] for a in poly])
                    
                    self.floors[idx]['rooms'].append({
                        'id': room_ID,
                        'name': room_name,
                        'bound': poly.T[::-1,:]
                    })
                if 'id' not in e.attrs:
                    continue
                if e.attrs['id'] == 'Stairs':
                    bounds = []
                    for child in e:
                        bound = child.polygon.attrs["points"]
                        bound = bound.split(' ')[:-1]
                        bound = np.array([list(map(np.float32, a.split(','))) for a in bound])
                        bounds.append(bound.T[::-1,:])
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
            self.floors[idx]['rooms'] = self.__mergeRoom(self.floors[idx]['rooms'])

            floorplans[idx]['style'] = ""
            img = self.__getImage("crispEdges")
            img = np.array(Image.fromarray(img))[:,:,:3]
            
            for i,c in enumerate(color):
                mask = (img== np.array(c)).all(-1)
                F = mask.astype(np.uint16)

                poly = (s for i, (s, v) in enumerate(shapes(F, mask=mask)))

                for l in poly:
                    pts = np.array(l['coordinates'][0], dtype=np.int32)
                    pts = np.apply_along_axis(lambda a: [1, *a], 1, pts)[:,::-1].T
                    
                    self.floors[idx]['boundaries'].append({
                        'id': i,
                        'name': Mapper.getBoundaryName(i),
                        'bound': pts
                    })
            
            floorplans[idx]['style'] = "display: none;"
        self.__houseFloorVisibility(True)


    def __getFurnitureBoundingBox(self, furniture):
        bound = furniture.find("g", {"class": "BoundaryPolygon"}).find("polygon")
        if bound is None or not bound.has_attr('points'):
            return
        bound = bound["points"]
        bound = bound.split(' ')[:4]
        bound = np.array([list(map(np.float32, a.split(','))) for a in bound])
        bound = np.vstack((
            bound.T[::-1,:],
            np.array([1, 1, 1, 1])
        ))

        matrix = furniture['transform']
        matrix = matrix[matrix.find("(") + 1 : matrix.find(")")]
        matt = list(map(np.float32, matrix.split(",")))
        matrix = np.array([
            [matt[3], matt[1], matt[5]],
            [matt[2], matt[0], matt[4]],
            [.0, .0, 1.0]
        ])

        bound = np.dot(matrix, bound)

        if furniture.parent["class"] == "FixedFurnitureSet":
            matrix = furniture.parent['transform']
            matrix = matrix[matrix.find("(") + 1 : matrix.find(")")]
            matt = list(map(np.float32, matrix.split(",")))
            matrix = np.array([
                [matt[3], matt[1], matt[5]],
                [matt[2], matt[0], matt[4]],
                [.0, .0, 1.0]
            ])

            bound = np.dot(matrix, bound)

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


    def __createImage(self):
        img = self.__getImage()
        
        transformation = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])

        if self.background is not None:
            img, transformation = remakeImage(
                img,
                self.background,
                scale=(1.3, 1.5),
                shift=(0.1, 0),
                W=fix_width
            )
        else:
            img = img[:,:,:3]

        img = qualityReduction(img)
        img = addNoise(img, 'poisson')

        return img, transformation


    def __saveImage(self, img, directory, resize_dim=image_dim, bg='resize'):
        if bg == 'resize':
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
                floorPoly = floor['polygon']
                value = np.vstack((floorPoly, np.ones((floorPoly.shape[1]))))
                transformed = np.dot(transformation, value)[:2]
                transformed[0,:] = transformed[0,:] / shape[0]
                transformed[1,:] = transformed[1,:] / shape[1]
                transformed = transformed.T.flatten()

                newline = '\n'
                if first:
                    newline = ''
                strg = f"{newline}0 {' '.join(np.char.mod('%f', transformed))}"
                first = False

                f.write(strg)


    def __saveBoundingBox(self, transformation, polygons, floorBound, directory):
        shape = (floorBound[2] - floorBound[0], floorBound[3] - floorBound[1])
        mx = max(shape[0], shape[1])
        with open(directory, "w") as f:
            first = True
            for furniture in polygons:
                ID = furniture['id']
                bound = furniture['bound']

                transformed = np.dot(transformation, bound)[:2]
                transformed[0,:] = (transformed[0,:] - floorBound[0] + (mx - shape[0]) // 2) / mx
                transformed[1,:] = (transformed[1,:] - floorBound[1] + (mx - shape[1]) // 2) / mx
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


    def setBackground(self, background):
        if not isinstance(background, str):
            return

        background_directory = f'{self.asset_directory}\\background\\{background}.jpg'

        if os.path.exists(os.path.dirname(background_directory)):
            s = openRgb(background_directory)
            self.background = s


    def setStyle(self, style, color=None):
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


    def getFloorplanCount(self):
        return len(self.floors)


    def saveStructure(self, target_directory):
        img, transformation = self.__createImage()
        tmp_img = Image.fromarray(img.astype(np.uint8))
        
        fileIDroi = nanoid.generate(namechar, 24)
        roi_image_dir = f'{target_directory}/roi-detection/image/{fileIDroi}.image.generated.png'
        roi_poly_dir = f'{target_directory}/roi-detection/label/{fileIDroi}.label.generated.txt'
        self.__saveImage(img, roi_image_dir)
        self.__savePoly(transformation, img.shape, roi_poly_dir)

        for floor in self.floors:
            # crop gambar, floor ini dapetnya polygon sebelum transformasi
            floorPoly = floor['polygon']
            value = np.vstack((floorPoly, np.ones((floorPoly.shape[1]))))
            transformed = np.dot(transformation, value)[:2].astype(np.int32)

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
            symbol_image_dir = f'{target_directory}/symbol-detection/image/{fileIDsymbol}.input.generated.png'
            symbol_poly_dir = f'{target_directory}/symbol-detection/label/{fileIDsymbol}.label.generated.txt'

            self.__saveImage(cropped, symbol_image_dir, bg='fill')
            self.__saveBoundingBox(transformation, floor['furnitures'] + floor['boundaries'], [top, left, bottom, right], symbol_poly_dir)

            # save graph
            fileIDroom = nanoid.generate(namechar, 24)
            room_image_dir = f'{target_directory}/room-classification/graph/{fileIDroom}.input.generated.pt'

            self.__saveGraph(floor, room_image_dir)


        # self.__saveFloorsPoly(
        #     transformation,
        #     f'{target_directory}/floors_poly/{fileID}.floors_poly.png'
        # )

# perlu diganti

    def extractFixedFurnitureBB(self, floor, parent):
        furnitures = floor.select(".FixedFurniture")
        
        for furniture in furnitures:
            bound = furniture.find("g", {"class": "BoundaryPolygon"}).find("polygon")
            if bound is None or not bound.has_attr('points'):
                continue
            bound = bound["points"]
            bound = bound.split(' ')[:4]
            bound = np.array([list(map(np.float32, a.split(','))) for a in bound])
            bound = np.vstack((
                bound.T[::-1,:],
                np.array([1, 1, 1, 1])
            ))

            matrix = furniture['transform']
            matrix = matrix[matrix.find("(") + 1 : matrix.find(")")]
            matt = list(map(np.float32, matrix.split(",")))
            matrix = np.array([
                [matt[3], matt[1], matt[5]],
                [matt[2], matt[0], matt[4]],
                [.0, .0, 1.0]
            ])

            bound = np.dot(matrix, bound)

            if furniture.parent["class"] == "FixedFurnitureSet":
                matrix = furniture.parent['transform']
                matrix = matrix[matrix.find("(") + 1 : matrix.find(")")]
                matt = list(map(np.float32, matrix.split(",")))
                matrix = np.array([
                    [matt[3], matt[1], matt[5]],
                    [matt[2], matt[0], matt[4]],
                    [.0, .0, 1.0]
                ])

                bound = np.dot(matrix, bound)
            
            bound = np.sort(bound.astype(np.uint16))[:,0::3]

            key = furniture["class"].split(" ")[1]
            key = Mapper.getIconName(key)

            self.insertBoundingBox(key, parent, bound)
            
    def extractStairsBB(self, floor, parent):
        stairs = floor.select(".Stairs")
        
        for stair in stairs:
            bound = stair.find("g", {"class": "Flight"}).find("polygon")
            if bound is None or not bound.has_attr('points'):
                continue
            bound = bound["points"]
            bound = bound.split(' ')[:4]
            bound = np.array([list(map(np.float32, a.split(','))) for a in bound])
            bound = np.vstack((
                bound.T[::-1,:],
                np.array([1, 1, 1, 1])
            ))

            bound = np.sort(bound.astype(np.uint16))[:,0::3]
            self.insertBoundingBox("Stairs", parent, bound)
    
    def extractBoundaries(self):
        def PolyArea(x,y):
            return .5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        self.setStyle('boundaries')
        
        floorplans = self.structure.select("g[class=Floor]")
        
        self.__houseFloorVisibility(False)
        
        index = 0
        
        label = ['', 'Wall', 'Door', 'Window', 'Floor']
        
        for floorplan in floorplans:
            floorplan['style'] = ""
            img = self.__getImage("crispEdges")
            img = Image.fromarray(img).convert('P', palette=Image.Palette.ADAPTIVE, colors=5)
            img = np.array(img)
            total = img.shape[0] * img.shape[1]
            
            for i in range(1,5):
                F = (img == i).astype(np.int32)

                poly = (s for i, (s, v) in enumerate(shapes(F)))

                for l in poly:
                    pts = np.array(l['coordinates'][0], dtype=np.int32)
                    area = PolyArea(pts[:,0], pts[:,1])
                    pts = np.apply_along_axis(lambda a: [1, *a], 1, pts)[:,::-1].T
                    if 2 * area < total:
                        self.insertBoundary(label[i], index, pts)
            
            floorplan['style'] = "display: none;"
            
            index += 1
            
        self.__houseFloorVisibility(True)
    
    def insertBoundingBox(self, key, parent, bounding_box):
        if key is None:
            return
        dct_key = (parent, key)
        if dct_key not in self.bounding_boxes:
            self.bounding_boxes[dct_key] = []
        self.bounding_boxes[dct_key].append(bounding_box)
    
    def insertBoundary(self, key, parent, boundary):
        if key is None:
            return
        dct_key = (parent, key)
        if dct_key not in self.boundaries:
            self.boundaries[dct_key] = []
        self.boundaries[dct_key].append(boundary)