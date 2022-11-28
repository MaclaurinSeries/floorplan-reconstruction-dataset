import sys, getopt, os, io
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from cairosvg import svg2png
from ImageUtil import remakeImage, openRgb, getFloorplanPolygon, qualityReduction, addNoise
import HouseConfig as Mapper
from PIL import Image
from rasterio.features import shapes
import cv2
import json
import nanoid.generate

__all__ = ['FloorPlanSVG']

namechar = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

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

        self.__preprocessing()


    def __preprocessing(self):
        for text_tag in self.structure.select('text'):
            text_tag.extract()

        self.__extractFloor()


    def __extractFloor(self):
        self.setStyle('mask')
        self.__houseFloorVisibility(True)
        self.floors = getFloorplanPolygon(
            self.__getImage('crispEdges')
        )


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


    def __saveImage(self, directory):
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
                W=(1000 + np.random.randint(40) * 10)
            )
        else:
            img = img[:,:,:3]

        img = qualityReduction(img)
        img = addNoise(img, 'poisson')

        stat = cv2.imwrite(directory, img[:,:,::-1])

        # return transformation


    def __saveFloorsPoly(self, transformation, directory):
        for floor_bounding_box in self.floors:
            value = np.vstack((floor_bounding_box, np.ones((floor_bounding_box.shape[1]))))
            transformed = np.dot(transformation, value).astype(int).T[:,:2].reshape((-1,1,2))

        return


    def __preprocessGraph(self, fileID):
        raise NotImplementedError
        return


    def setBackground(self, background):
        if not isinstance(background, str):
            return

        background_directory = f'{self.asset_directory}\\background\\{background}.jpg'

        if os.path.exists(os.path.dirname(background_directory)):
            s = openRgb(background_directory)
            self.background = s


    def setStyle(self, style):
        if not isinstance(style, str):
            return
        
        style_directory = f'{self.asset_directory}\\style\\{style}.css'
        
        if os.path.exists(os.path.dirname(style_directory)):
            s = ""
            
            with open(style_directory) as f:
                s = f.read()
        
            for style_tag in self.structure.select('style'):
                style_tag.extract()

            tag = self.structure.new_tag("style")
            tag.string = s
            self.structure.svg.g.insert_before(tag)


    def getFloorplanCount(self):
        return len(self.floors)


    def saveStructure(self, target_directory='./__collection__'):
        fileID = nanoid.generate(namechar, 16)

        transformation = self.__saveImage(
            f'{target_directory}/roi-detection/image/{fileID}.input.generated.png'
        )
        # self.__saveFloorsPoly(
        #     transformation,
        #     f'{target_directory}/floors_poly/{fileID}.floors_poly.png'
        # )


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
            
            bound = np.sort(bound.astype(np.uint32))[:,0::3]

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

            bound = np.sort(bound.astype(np.uint32))[:,0::3]
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