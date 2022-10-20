import sys, getopt, os, io
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from cairosvg import svg2png
from image_util import remake_image, open_rgb, open_rgba, get_bounding_box
import cv2

import matplotlib.pyplot as plt

def main(argv):
    input_directory = '.\\input'
    output_directory = '.\\output'
    asset_directory = '.\\asset'
    
    help_text = 'generator.py -i <input_directory> -o <output_directory> -a <asset_directory>'
    
    try:
        opts, args = getopt.getopt(argv,"h:i:o:a:",[
                                   "help=",
                                   "input-dir=",
                                   "output-dir=",
                                   "asset-dir="])
    except getopt.GetoptError:
        print(help_text)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_text)
            sys.exit()
        elif opt in ("-i", "--input-dir"):
            input_directory = arg
        elif opt in ("-o", "--output-dir"):
            output_directory = arg
        elif opt in ("-s", "--asset-dir"):
            asset_directory = arg
    
    for folder in os.scandir(input_directory):
        if folder.is_dir():
            for filename in os.scandir(folder.path):
                if filename.is_file() and filename.name.endswith(".svg"):
                    process_image(filename.name, asset_directory, output_directory)
        elif folder.is_file() and folder.name.endswith(".svg"):
            process_image(folder.name, asset_directory, output_directory)

def process_image(os_filename, asset_dir, out_dir):
    # style_number = np.random.randint(0, 6)
    style_number = 6
    
    bounding_box = {}
    
    with open(os_filename) as fsvg:
        svg = fsvg.read()
        with open(f'{asset_dir}\\style\\style_{style_number}.css') as fstyle:
            
            soup = BeautifulSoup(svg, "xml")
            
            floorplans = soup.select("g[class=Floor]")

            if (len(floorplans) > 2):
                return None
            
            tag = soup.new_tag("style")
            tag.string = fstyle.read()
            soup.svg.g.insert_before(tag)

            for floorplan in floorplans:
                for s in floorplan.select('text'):
                    s.extract()
                
                floorplan['style'] = ""
                
                svg = str(soup)
                bb = extract_BB(svg)
                
                bounding_box[floorplan.g.get('id')] = bb
                
                floorplan['style'] = "display: none;"
            
            for floorplan in floorplans:
                floorplan['style'] = ""
                
        svg = str(soup)
        
    mem = io.BytesIO()
    svg2png(bytestring=svg, write_to=mem)
    img = np.array(Image.open(mem))
    
    bg = open_rgb(f'{asset_dir}\\background\\bg_{style_number}.jpg')
    nimg, translation = remake_image(img, bg, scale=(1.3, 1.5), shift=(0.1, 0))
    
    # saving image with bounding box
    for key,value in bounding_box.items():
        trans = np.dot(translation, value).astype(int)
        bounding_box[key] = trans
        trans = trans.T
        cv2.rectangle(nimg, trans[0], trans[1], (0,0,255), 1)
        cv2.putText(nimg, key, trans[0], 0, 0.5, (0,0,255), 1//3)
        
    cv2.imwrite('saved.png', nimg[:,:,::-1])

def extract_BB(svg):
    mem = io.BytesIO()
    svg2png(bytestring=svg, write_to=mem)
    img = np.array(Image.open(mem))
    
    xl, yl, xr, yr = get_bounding_box(img)
    
#     return [
#         [yr, yl, yl, yr, yr],
#         [xl, xl, xr, xr, xl],
#         [1, 1, 1, 1, 1]
#     ]
    return [
        [yl, yr],
        [xl, xr],
        [1, 1]
    ]

def apply_style(svg, style):
    with open(style) as f:
        soup = BeautifulSoup(svg, "xml")
        
        if (len(soup.find_all("div", {"class": "Floorplan"})) > 2):
            return None

        for s in soup.select('text'):
            s.extract()
        
        tag = soup.new_tag("style")
        tag.string = f.read()
        soup.svg.g.insert_before(tag)
        
        soup.svg.g.g.attrs["filter"] = "url(#texture)"
        
        svg = str(soup)
        
    return svg
            
if __name__ == "__main__":
    main(sys.argv[1:])