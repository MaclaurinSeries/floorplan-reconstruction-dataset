import sys, getopt, os, io
from FloorPlanSVG import FloorPlanSVG

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
    
    floorplan = FloorPlanSVG(f'.\\{os_filename}', asset_dir)
    floorplan.set_style(f'style_{style_number}')
    floorplan.set_background(f'bg_{style_number}')
    floorplan.save_image()
            
if __name__ == "__main__":
    main(sys.argv[1:])