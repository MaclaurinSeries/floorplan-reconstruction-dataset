import sys, getopt, os, io
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from cairosvg import svg2png
from image_util import remake_image, open_rgb, open_rgba, get_floorplan_bounding_box
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
        
        self.set_style('mask')
        
        for text_tag in self.structure.select('text'):
            text_tag.extract()
        
        self.bounding_boxes = {}
        
        self.extract_floor_BB()
        self.extract_fixed_furniture_BB()
        self.extract_stairs_BB()
    
    def set_background(self, background):
        if not isinstance(background, str):
            return
        
        background_directory = f'{self.asset_directory}\\background\\{background}.jpg'
        
        if os.path.exists(os.path.dirname(background_directory)):
            s = open_rgb(background_directory)
            self.background = s
            self.selected_background = background
    
    def set_style(self, style):
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
        
    def get_floorplan_count(self):
        floorplans = self.structure.select("g[class=Floor]")
        return len(floorplans)
    
    def get_image(self):
        svg = str(self.structure)
        
        mem = io.BytesIO()
        svg2png(bytestring=svg, write_to=mem)
        
        return np.array(Image.open(mem))
    
    def get_bounding_boxes(self, json=False, outfile=None):
        if json:
            return json.dumps(self.bounding_boxes, indent=4)
        if outfile is None:
            return self.bounding_boxes
        
        with open(outfile, "w") as f:
            json.dump(self.bounding_boxes, f)
    
    def save_image(self, target_directory='saved.png', with_bounding_box=True):
        img = self.get_image()
        cv2.imwrite(target_directory, img)
        transformation = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        
        if self.background is not None:
            img, transformation = remake_image(
                img,
                self.background,
                scale=(1.3, 1.5),
                shift=(0.1, 0)
            )
        
        for key,bounding_box_list in self.bounding_boxes.items():
            for value in bounding_box_list:
                
                transformed = np.dot(transformation, value).astype(int).T
                
                cv2.rectangle(img, transformed[0], transformed[1], (0,0,255), 1)
                cv2.putText(img, key, transformed[0], 0, 1, (0,0,255), 1)

        cv2.imwrite(target_directory, img[:,:,::-1])
    
    def extract_floor_BB(self):
        floorplans = self.structure.select("g[class=Floor]")
        
        self.house_floor_visibility(False)
            
        for floorplan in floorplans:
            floorplan['style'] = ""
            img = self.get_image()
            
            xl, yl, xr, yr = get_floorplan_bounding_box(img)
            
            self.insert_bounding_box('Floor', np.array([
                [yl, yr],
                [xl, xr],
                [1, 1]
            ]))
            floorplan['style'] = "display: none;"
            
        self.house_floor_visibility(True)
    
    def extract_fixed_furniture_BB(self):
        furnitures = self.structure.select(".FixedFurniture")
                
        for furniture in furnitures:
            bound = furniture.find("g", {"class": "BoundaryPolygon"}).find("polygon")
            if bound is None or not bound.has_attr('points'):
                continue
            bound = bound["points"]
            bound = bound.split(' ')[:4]
            bound = [list(map(float, (a + ",1").split(','))) for a in bound]
            bound.sort()
            bound = np.array([
                bound[0],
                bound[1],
                bound[2],
                bound[3]
            ]).T

            matrix = furniture['transform']
            matrix = matrix[matrix.find("(") + 1 : matrix.find(")")]
            matt = list(map(float, matrix.split(",")))
            matrix = np.array([
                [matt[0], matt[2], matt[4]],
                [matt[1], matt[3], matt[5]],
                [.0, .0, 1.0]])

            bound = np.dot(matrix, bound)

            if furniture.parent["class"] == "FixedFurnitureSet":
                matrix = furniture.parent['transform']
                matrix = matrix[matrix.find("(") + 1 : matrix.find(")")]
                matt = list(map(float, matrix.split(",")))
                matrix = np.array([
                    [matt[0], matt[2], matt[4]],
                    [matt[1], matt[3], matt[5]],
                    [.0, .0, 1.0]])

                bound = np.dot(matrix, bound)

            bound = np.sort(bound.astype(int))[:,1:3]

            key = furniture["class"].split(" ")
            key.remove("FixedFurniture")
            key = " ".join(key)

            self.insert_bounding_box(key, bound)
            
    def extract_stairs_BB(self):
        stairs = self.structure.select(".Stairs")
                
        for stair in stairs:
            bound = stair.find("g", {"class": "Flight"}).find("polygon")
            if bound is None or not bound.has_attr('points'):
                continue
            bound = bound["points"]
            bound = bound.split(' ')[:4]
            bound = [list(map(float, (a + ",1").split(','))) for a in bound]
            bound.sort()
            bound = np.array([
                bound[0],
                bound[1],
                bound[2],
                bound[3]
            ]).T

            bound = np.sort(bound.astype(int))[:,1:3]

            self.insert_bounding_box("Stairs", bound)
    
    def house_floor_visibility(self, visible=True):
        floorplans = self.structure.select("g[class=Floor]")
        
        display = "display: none;"
        if visible: display = ""
        
        for floorplan in floorplans:
            floorplan['style'] = display
    
    def insert_bounding_box(self, key, bounding_box):
        if key not in self.bounding_boxes:
            self.bounding_boxes[key] = []
        self.bounding_boxes[key].append(bounding_box)