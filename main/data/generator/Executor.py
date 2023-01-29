import sys, getopt, os, io
import argparse
from FloorPlanSVG import FloorPlanSVG

def main(args):
    prepare_directory(args.output_directory)

    for folder in os.scandir(args.input_directory):
        if folder.is_dir():
            for filename in os.scandir(folder.path):
                if filename.is_file() and filename.name.endswith(".svg"):
                    process_image(filename.name, args.asset_directory, args.output_directory)
        elif folder.is_file() and folder.name.endswith(".svg"):
            process_image(folder.name, args.asset_directory, args.output_directory)


def process_image(os_filename, asset_dir, out_dir):
    # style_number = np.random.randint(0, 6)
    
    style_number = 6
    
    floorplan = FloorPlanSVG(f'.\\{os_filename}', asset_dir)
    floorplan.setStyle(f'style_{style_number}')
#     floorplan.setStyle('boundaries')
    floorplan.setBackground(f'bg_{style_number}')
    floorplan.saveStructure(out_dir)


def prepare_directory(directory):
    paths = [
        f'{directory}\\roi-detection\\image',
        f'{directory}\\roi-detection\\label',
        f'{directory}\\room-classification\\graph',
        f'{directory}\\symbol-detection\\image',
        f'{directory}\\symbol-detection\\label'
    ]

    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'DataSet Generator',
        description = 'This program augment and create new dataset. Only for internal usage.',
        epilog = 'Created by Christian Budhi Sabdana aka MaclaurinSeries (GitHub)'
    )
    parser.add_argument('-i', '--input-dir',
                        type=str,
                        default='.\\input',
                        help='dataset input directory',
                        dest='input_directory')
    parser.add_argument('-o', '--output-dir',
                        type=str,
                        default='.\\output',
                        help='directory to place reproduced data',
                        dest='output_directory')
    parser.add_argument('-a', '--asset-dir',
                        type=str,
                        default='.\\asset',
                        help='directory for random property to image',
                        dest='asset_directory')
    
    main(parser.parse_args())