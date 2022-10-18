import sys, getopt, os
from bs4 import BeautifulSoup
from cairosvg import svg2png
from image_util import remake_image, open_rgb, open_rgba

def main(argv):
    input_directory = 'input'
    output_directory = '.'
    style_directory = 'style.css'
    background_directory = 'bg'
    texture_directory = 'txt'
    
    help_text = 'generator.py -i <input_directory> -o <output_directory> -s <style_directory> -b <background_directory> -t <texture_directory>'
    
    try:
        opts, args = getopt.getopt(argv,"h:i:o:s:b:t:",["help=","idir=","odir=","style=","background=","texture="])
    except getopt.GetoptError:
        print(help_text)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(help_text)
            sys.exit()
        elif opt in ("-i", "--idir"):
            input_directory = arg
        elif opt in ("-o", "--odir"):
            output_directory = arg
        elif opt in ("-s", "--style"):
            style_directory = arg
        elif opt in ("-b", "--background"):
            style_directory = arg
        elif opt in ("-t", "--texture"):
            style_directory = arg
        
    
    for folder in os.scandir(input_directory):
        stop = False
        if folder.is_dir():
            for filename in os.scandir(folder.path):
                if filename.is_file() and filename.name.endswith(".svg"):
                    reformat_file(filename.path, style_directory, output_directory, os.path.splitext(filename.name)[0])
                    stop = True
        elif folder.is_file() and folder.name.endswith(".svg"):
            reformat_file(folder.path, style_directory, output_directory, os.path.splitext(folder.name)[0])
            stop = True
            
        if stop: break
        
def apply_style(svg, style):
    with open(style) as f:
        soup = BeautifulSoup(svg, "xml")

        for s in soup.select('text'):
            s.extract()
        
        tag = soup.new_tag("style")
        tag.string = f.read()
        soup.svg.g.insert_before(tag)
        
        soup.svg.g.g.attrs["filter"] = "url(#texture)"
        
        if (len(soup.find_all("div", {"class": "stylelistrow"})) >= 3) 
        
        svg = str(soup)
        
    return svg
    
def reformat_file(image_path, style_dir, output_dir, filename):
    with open(image_path) as f:
        svg = apply_style(
            f.read(),
            style_dir)
    
    svg2png(bytestring = svg, write_to = f'{output_dir}\\{filename}.png')
            
if __name__ == "__main__":
    main(sys.argv[1:])