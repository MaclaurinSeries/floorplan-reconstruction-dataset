import sys, getopt, os, io, warnings
import argparse
import numpy as np
import traceback, logging
from FloorPlanSVG import FloorPlanSVG
import concurrent.futures as parallel
from datetime import datetime

def main(args, limiter=None):
    prepare_directory(args.output_directory)
    
    selected_image = []
    if args.selected_image_dir is not None:
        with open(args.selected_image_dir, "r") as f:
            selected_image = f.read().split('\n')

    function_arguments = []
    arg_id = 0

    if limiter is not None:
        iterator = limiter
    for directory in args.input_directory:
        for folder in os.scandir(directory):
            if folder.is_dir():
                save_image = folder.name in selected_image

                for filename in os.scandir(folder.path):
                    if filename.is_file() and filename.name.endswith(".svg"):
                        function_arguments.append(dict(
                            os_filename=f'{folder.path}\\{filename.name}',
                            asset_dir=args.asset_directory,
                            out_dir=args.output_directory,
                            save_image=save_image,
                            ID=folder.name
                        ))
                        arg_id += 1
            elif folder.is_file() and folder.name.endswith(".svg"):
                function_arguments.append(dict(
                    os_filename=f'{folder.path}',
                    asset_dir=args.asset_directory,
                    out_dir=args.output_directory,
                    ID=arg_id
                ))
                arg_id += 1
            if limiter is not None:
                iterator -= 1
                if iterator <= 0:
                    break
        if limiter is not None and iterator <= 0:
            break
    
    START_TIME = datetime.now()
    LOG_FILE = f'run_{START_TIME.strftime("%Y-%m-%dT%H%M%S")}.log'

    success_count = 0
    failed_count = 0
    
    print(f'executed at {START_TIME.strftime("%Y-%m-%d %H:%M:%S")}')

    with parallel.ProcessPoolExecutor(max_workers=8) as x:
        for idx,(result, success) in enumerate(x.map(parallel_process, function_arguments)):
            with open(LOG_FILE, 'a') as f:
                f.write(result)
            if success: success_count += 1
            else: failed_count += 1
        print(f'result:\n\t{str(success_count).rjust(5," ")} success, {str(failed_count).rjust(5," ")} failed\n\truntime {(datetime.now() - START_TIME).total_seconds()} seconds')


def parallel_process(args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result, success = process_image(**args)
    return result, success


def process_image(os_filename, asset_dir, out_dir, ID, save_image=False):
    style_number = np.random.randint(0, 14) + 1

    log_result = ''
    success = True

    try:
        floorplan = FloorPlanSVG(os_filename, asset_dir, ID)
        floorplan.setStyle(f'style_{style_number}')
        floorplan.setBackground(f'bg_{style_number}')
        floorplan.saveStructure(out_dir)

        if save_image:
            floorplan = FloorPlanSVG(f'.\\{os_filename}', asset_dir, -1)
            floorplan.saveOriginalImage(f'.\\F1_scaled.png', out_dir)
    except Exception as e:
        logging.info(f'ERROR on file {ID}')
        log_result = f'ERROR FILE {ID}\n   LOCATION: {os_filename}\n        LOG: {e}\n  TRACEBACK:\n{traceback.format_exc()}'
        success = False
    return log_result, success


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
                        nargs='+',
                        default=['.\\input'],
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
    parser.add_argument('-s', '--selected',
                        type=str,
                        default=None,
                        help='txt file containing id of selected image',
                        dest='selected_image_dir')
    
    main(parser.parse_args())