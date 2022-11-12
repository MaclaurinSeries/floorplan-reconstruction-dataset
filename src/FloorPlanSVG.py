import sys, getopt, os, io
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from cairosvg import svg2png
from ImageUtil import remakeImage, openRgb, getFloorplanBoundingBox
from PIL import Image
from rasterio.features import shapes
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import cv2
import json

class FloorPlanSVG:
    def __init__(self, svg, asset_dir):
        s = svg
        
        if os.path.exists(os.path.dirname(svg)):
            with open(svg) as f:
                s = f.read()
            
        self.structure = BeautifulSoup(s, "xml")
        self.background = None
        self.asset_directory = asset_dir
        self.selected_style = None
        self.selected_background = None
        
        for text_tag in self.structure.select('text'):
            text_tag.extract()
        
        self.structure.svg.attrs['shape-rendering'] = "crispEdges"
        
        self.bounding_boxes = {}
        self.boundaries = {}
        
        self.extractFloorBB()
        self.extractBoundaries()
    
    def setBackground(self, background):
        if not isinstance(background, str):
            return
        
        background_directory = f'{self.asset_directory}\\background\\{background}.jpg'
        
        if os.path.exists(os.path.dirname(background_directory)):
            s = openRgb(background_directory)
            self.background = s
            self.selected_background = background
    
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
            self.selected_style = style
        
    def getFloorplanCount(self):
        floorplans = self.structure.select("g[class=Floor]")
        return len(floorplans)
    
    def getImage(self):
        svg = str(self.structure)
        
        mem = io.BytesIO()
        svg2png(bytestring=svg, write_to=mem)
        
        return np.array(Image.open(mem))
    
    def getBoundingBoxes(self, json=False, outfile=None):
        if json:
            return json.dumps(self.bounding_boxes, indent=4)
        if outfile is None:
            return self.bounding_boxes
        
        with open(outfile, "w") as f:
            json.dump(self.bounding_boxes, f)
    
    def saveImage(self, target_directory='saved.png', with_bounding_box=True, with_boundaries=True):
        self.structure.svg.attrs.pop('shape-rendering')
        img = self.getImage()
        self.structure.svg.attrs['shape-rendering'] = "crispEdges"
        
        transformation = np.array([
            [1, 0, 0],
            [0, 1, 0]
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
        
        if not with_bounding_box and not with_boundaries:
            cv2.imwrite(target_directory, img)
            
        if with_bounding_box:
            for key,bounding_box_list in self.bounding_boxes.items():
                for value in bounding_box_list:

                    transformed = np.dot(transformation, value).astype(int).T

                    cv2.rectangle(img, transformed[0,1::-1], transformed[1,1::-1], (0,0,255), 1)
                    cv2.putText(img, key[1] + "-" + str(key[0]), transformed[0,1::-1], 0, 1, (0,0,255), 1)
        if with_boundaries:
            color = {
                'Wall': (255, 0, 0),
                'Door': (0, 255, 0),
                'Window': (0, 0, 0),
                'Floor': (0, 0, 255),
            }
            for key,boundary in self.boundaries.items():
                for value in boundary:
                    
                    transformed = np.dot(transformation, value).astype(int).T

                    cv2.polylines(img, [transformed[:,1::-1]], True, color[key[1]], 2)
        cv2.imwrite(target_directory, img[:,:,::-1])
    
    def extractFloorBB(self):
        self.setStyle('mask')
        
        floorplans = self.structure.select("g[class=Floor]")
        
        self.houseFloorVisibility(False)
        
        index = 0
            
        for floorplan in floorplans:
            floorplan['style'] = ""
            img = self.getImage()
            
            top_left, bottom_right = getFloorplanBoundingBox(img)
            
            self.insertBoundingBox('Floor', index, np.array([
                [top_left[0], bottom_right[0]],
                [top_left[1], bottom_right[1]],
                [1, 1],
            ]))
            self.extractFixedFurnitureBB(floorplan, index)
            self.extractStairsBB(floorplan, index)
            floorplan['style'] = "display: none;"
            
            index += 1
            
        self.houseFloorVisibility(True)
    
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
                [.0, .0, 1.0]])

            bound = np.dot(matrix, bound)

            if furniture.parent["class"] == "FixedFurnitureSet":
                matrix = furniture.parent['transform']
                matrix = matrix[matrix.find("(") + 1 : matrix.find(")")]
                matt = list(map(np.float32, matrix.split(",")))
                matrix = np.array([
                    [matt[3], matt[1], matt[5]],
                    [matt[2], matt[0], matt[4]],
                    [.0, .0, 1.0]])

                bound = np.dot(matrix, bound)
            
            bound = np.sort(bound.astype(np.uint32))[:,0::3]

            key = furniture["class"].split(" ")
            key.remove("FixedFurniture")
            key = " ".join(key)

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
        
        self.houseFloorVisibility(False)
        
        index = 0
        
        label = ['', 'Wall', 'Door', 'Window', 'Floor']
        
        for floorplan in floorplans:
            floorplan['style'] = ""
            img = self.getImage()
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
            
        self.houseFloorVisibility(True)
    
    def houseFloorVisibility(self, visible=True):
        floorplans = self.structure.select("g[class=Floor]")
        
        display = "display: none;"
        if visible: display = ""
        
        for floorplan in floorplans:
            floorplan['style'] = display
    
    def insertBoundingBox(self, key, parent, bounding_box):
        dct_key = (parent, key)
        if dct_key not in self.bounding_boxes:
            self.bounding_boxes[dct_key] = []
        self.bounding_boxes[dct_key].append(bounding_box)
    
    def insertBoundary(self, key, parent, boundary):
        dct_key = (parent, key)
        if dct_key not in self.boundaries:
            self.boundaries[dct_key] = []
        self.boundaries[dct_key].append(boundary)