import PySimpleGUI as sg
import cv2, os, io
import numpy as np
from PIL import Image
import torch
from ImageUtil import (
    openRgb
)

BASE_IMAGE_URL = f'./__collection__/symbol-detection/image'
BASE_GRAPH_URL = f'./__collection__/room-classification/graph'

def display_image(path, filename):
    img = openRgb(path)

    height = 600
    width = int(img.shape[1] * height / img.shape[0])
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    data = torch.load(f'{BASE_GRAPH_URL}/{filename.split(".")[0]}.pt')

    with open(f'./__collection__/symbol-detection/label/{filename.split(".")[0]}.txt') as f:
        lines = f.read().split('\n')
    
    pts = []
    for line in lines:
        numbers = list(map(float, line.split(' ')))

        cat = int(numbers[0])
        poly = np.array(numbers[1:])

        poly[0::2] *= height
        poly[1::2] *= width

        poly = poly.astype(np.int32)

        pts.append(np.array([np.clip(poly[1::2], 0,width - 1), np.clip(poly[0::2], 0,height - 1)]).T.reshape((-1, 1, 2)))

    img = cv2.polylines(img, pts, True, (255, 0, 0), 2)


    img = Image.fromarray(img[:,:,::-1])
    binary = io.BytesIO()
    img.save(binary, format='PNG')
    return binary.getvalue()

def main(viewer_path):
    sg.theme("LightGreen")

    done = []

    with open(viewer_path, "r") as f:
        done = f.read().split('\n')

    files = [f for f in os.listdir(BASE_IMAGE_URL) if os.path.isfile(f'{BASE_IMAGE_URL}/{f}') and f not in done]
    current = files.pop(0)

    # Define the window layout
    layout = [
        [
            sg.Text(current, size=(60, 1), justification="center", key='-FILENAME-'),
            sg.Text(f"DONE: {len(done)}", size=(60, 1), justification="center", key='-DONE-')
        ],
        [sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current}', current), key="-IMG-", size=(800,600))],
        [sg.Button("Next", size=(10, 1), key="-NEXT-")],
        [
            sg.InputText(key='-TEXT-', size=(40,1)),
            sg.Button("Save", size=(10, 1), key="-SAVE-")
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

        if event == '-NEXT-':
            if current not in done:
                done.append(current)
            if len(files) <= 0:
                sg.Popup('all done', keep_on_top=True)
            else:
                current = files.pop(0)
            window['-FILENAME-'].update(current)
            window['-DONE-'].update(f"DONE: {len(done)}")
            window['-IMG-'].update(data=display_image(f'{BASE_IMAGE_URL}/{current}', current))

    window.close()

main("viewer.log")