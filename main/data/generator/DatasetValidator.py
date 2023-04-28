import PySimpleGUI as sg
import cv2, os, io
import numpy as np
from PIL import Image
import torch
from ImageUtil import (
    openRgb
)

BASE_IMAGE_URL = f'./__collection__/symbol-detection/resized'
BASE_GRAPH_URL = f'./__collection__/symbol-detection/mask'

def display_image(path, filename):
    img = openRgb(path)
    mask = openRgb(f'./__collection__/symbol-detection/mask/{filename.split(".")[0]}.png')

    height = 200
    width = 200
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

    img = (np.array(img) * (np.array(mask)[:,:,None] // 255)).astype(np.uint8)

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
    current = []
    wh = (200,200)
    for i in range(9):
        current.append(files.pop(0))

    # Define the window layout
    layout = [
        [
            sg.Text(current, size=(90, 1), justification="center", key='-FILENAME-'),
            sg.Text(f"DONE: {len(done)}", size=(20, 1), justification="center", key='-DONE-')
        ],
        [
            [
                sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current[0]}', current[0]), key="-IMG0-", size=wh),
                sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current[1]}', current[1]), key="-IMG1-", size=wh),
                sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current[2]}', current[2]), key="-IMG2-", size=wh),
            ],
            [
                sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current[3]}', current[3]), key="-IMG3-", size=wh),
                sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current[4]}', current[4]), key="-IMG4-", size=wh),
                sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current[5]}', current[5]), key="-IMG5-", size=wh),
            ],
            [
                sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current[6]}', current[6]), key="-IMG6-", size=wh),
                sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current[7]}', current[7]), key="-IMG7-", size=wh),
                sg.Image(data=display_image(f'{BASE_IMAGE_URL}/{current[8]}', current[8]), key="-IMG8-", size=wh),
            ]
        ],
        [
            [
                sg.Checkbox("", key="-DELETE0-"),
                sg.Checkbox("", key="-DELETE1-"),
                sg.Checkbox("", key="-DELETE2-"),
            ],
            [
                sg.Checkbox("", key="-DELETE3-"),
                sg.Checkbox("", key="-DELETE4-"),
                sg.Checkbox("", key="-DELETE5-"),
            ],
            [
                sg.Checkbox("", key="-DELETE6-"),
                sg.Checkbox("", key="-DELETE7-"),
                sg.Checkbox("", key="-DELETE8-"),
            ],
        ],
        [
            sg.Text("[]", key='-TEXT-', size=(40,1)),
            sg.Button("Save", size=(10, 1), key="-NEXT-"),
            sg.Button("Skip", size=(10, 1), key="-SKIP-")
        ],
    ]

    # Create the window and show it without the plot
    window = sg.Window("Dataset Validator", layout, location=(100,0))
    deleted = [False] * 9

    while True:
        event, values = window.read(timeout=20)
        if event == sg.WIN_CLOSED:
            with open(viewer_path, 'w') as f:
                f.write('\n'.join(done))
            break

        for i in range(9):
            deleted[i] = values[f"-DELETE{i}-"]
        window['-TEXT-'].update(str([i for i in range(9) if deleted[i]]))
        

        if event == '-NEXT-':
            for i,c in enumerate(current):
                if c not in done and not deleted[i]:
                    done.append(c)
                window[f'-DELETE{i}-'].update(False)
            if len(files) < 9:
                sg.Popup('all done', keep_on_top=True)
            else:
                st = ""
                for i in range(9):
                    current[i] = files.pop(0)
                    window[f'-IMG{i}-'].update(data=display_image(f'{BASE_IMAGE_URL}/{current[i]}', current[i]))
                    st += current[i] + " "
                window['-FILENAME-'].update(st)
            window['-DONE-'].update(f"DONE: {len(done)}")
        elif event == '-SKIP-':
            for i in range(9):
                st = ""
                for i in range(9):
                    current[i] = files.pop(0)
                    window[f'-IMG{i}-'].update(data=display_image(f'{BASE_IMAGE_URL}/{current[i]}', current[i]))
                    st += current[i] + " "
                window['-FILENAME-'].update(st)

    window.close()

main("viewer1.log")