import PySimpleGUI as sg
import cv2, os, io, warnings
import numpy as np
from PIL import Image
import torch
from ImageUtil import (
    openRgb
)
from FloorPlanSVG import FloorPlanSVG


def display_image(filename, similarity_param):
    print(filename)
    [root_path] = [
        f'.\\cubicasa5k\\{a}\\{filename}' for a in ['colorful', 'high_quality', 'high_quality_architectural'] if os.path.exists(f'.\\cubicasa5k\\{a}\\{filename}')
    ]
    svg_path = f'{root_path}\\model.svg'
    img_path = f'{root_path}\\F1_original.png'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        floorplan = FloorPlanSVG(svg_path, '.\\asset', filename)
        img, transformation = floorplan.displayOriginalImage(img_path, similarity_param)
    
    if img is None:
        img = openRgb(img_path)
        sg.popup('invalid value', keep_on_top=True)

    height = 400
    width = int(img.shape[1] * height / img.shape[0])
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    img = Image.fromarray(img[:,:,::-1])
    binary = io.BytesIO()
    img.save(binary, format='PNG')
    return binary.getvalue(), transformation, svg_path, img_path

def main(viewer_path, used_path):
    sg.theme("LightGreen")

    done = []

    with open(viewer_path, "r") as f:
        done = f.read().split('\n')

    files = []

    with open(used_path, "r") as f:
        files = [a for a in f.read().split('\n') if a not in done]

    current = files.pop(0)
    param = dict(
        min_matches=50,
        sift_count=600,
        trees=64,
        checks=20,
        distance_factor=0.9,
        ransac_factor=5.0
    )

    img_bytes, transformation, svg_path, img_path = display_image(current, param)

    # Define the window layout
    layout = [
        [
            sg.Text(current, size=(60, 1), justification="center", key='-FILENAME-'),
            sg.Text(f"DONE: {len(done)}", size=(60, 1), justification="center", key='-DONE-')
        ],
        [sg.Image(data=img_bytes, key="-IMG-", size=(800,400))],
        [
            sg.Button("View", size=(10, 1), key="-VIEW-"),
            sg.Button("Save", size=(10, 1), key="-SAVE-"),
            sg.Button("Skip", size=(10, 1), key="-SKIP-"),
        ],
        [
            sg.Text('min_matches', size=(20, 1), justification="right"),
            sg.Slider(range=(10,500), orientation="h", resolution=1, default_value=param['min_matches'], key='-INPUT-MIN-MATCHES-', size=(80,10))
        ],
        [
            sg.Text('sift_count', size=(20, 1), justification="right"),
            sg.Slider(range=(600,20000), orientation="h", resolution=100, default_value=param['sift_count'], key='-INPUT-SIFT-COUNT-', size=(80,10))
        ],
        [
            sg.Text('trees', size=(20, 1), justification="right"),
            sg.Slider(range=(4,256), orientation="h", resolution=1, default_value=param['trees'], key='-INPUT-TREES-', size=(80,10))
        ],
        [
            sg.Text('checks', size=(20, 1), justification="right"),
            sg.Slider(range=(1,50), orientation="h", resolution=1, default_value=param['checks'], key='-INPUT-CHECKS-', size=(80,10))
        ],
        [
            sg.Text('distance_factor', size=(20, 1), justification="right"),
            sg.Slider(range=(0,1), orientation="h", resolution=0.01, default_value=param['distance_factor'], key='-INPUT-DISTANCE-FACTOR-', size=(80,10))
        ],
        [
            sg.Text('ransac_factor', size=(20, 1), justification="right"),
            sg.Slider(range=(0,50), orientation="h", resolution=0.1, default_value=param['ransac_factor'], key='-INPUT-RANSAC-FACTOR-', size=(80,10))
        ],
    ]

    # Create the window and show it without the plot
    window = sg.Window("Dataset Validator", layout, location=(300,50))


    while True:
        event, values = window.read(timeout=20)
        if event == sg.WIN_CLOSED:
            with open(viewer_path, 'w') as f:
                f.write('\n'.join(done))
            break

        if event == '-SKIP-':
            if current not in done:
                done.append(current)
            if len(files) <= 0:
                sg.Popup('all done', keep_on_top=True)
            else:
                current = files.pop(0)
            with open(viewer_path, 'w') as f:
                f.write('\n'.join(done))
            img_bytes, transformation, svg_path, img_path = display_image(current, param)
            window['-FILENAME-'].update(current)
            window['-DONE-'].update(f"DONE: {len(done)}")
            window['-IMG-'].update(data=img_bytes)
        if event == '-SAVE-':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                floorplan = FloorPlanSVG(svg_path, '.\\asset', current)
                floorplan.saveOriginalImage(img_path, './__collection__', None, transformation)
            if current not in done:
                done.append(current)
            if len(files) <= 0:
                sg.Popup('all done', keep_on_top=True)
            else:
                current = files.pop(0)
            with open(viewer_path, 'w') as f:
                f.write('\n'.join(done))
            img_bytes, transformation, svg_path, img_path = display_image(current, param)
            window['-FILENAME-'].update(current)
            window['-DONE-'].update(f"DONE: {len(done)}")
            window['-IMG-'].update(data=img_bytes)
        if event == '-VIEW-':
            for key in list(param.keys()):
                slider_key = '-'.join(['', 'INPUT', *key.upper().split('_'), ''])
                param[key] = values[slider_key]
            img_bytes, transformation, svg_path, img_path = display_image(current, param)
            
            window['-IMG-'].update(data=img_bytes)

    window.close()

main("viewer.log", ".\\cubicasa5k\\backup\\dataset_used_12Oct2022_16-42.txt")