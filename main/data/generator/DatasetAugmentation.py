import argparse, os
import numpy as np
import cv2
from scipy import ndimage
from ImageUtil import (
    openRgb
)

target_directory = './test-folder'

def color_space_augmentation(image, label):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random = np.random.random(2) + 0.5
    hsv[:,:,1] = np.clip(hsv[:,:,1] * random[0], a_min=0, a_max=255).astype(np.int8)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * random[1], a_min=0, a_max=255).astype(np.int8)
    
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return img, label


def rotate90(image, label):
    new_image = np.concatenate((np.rot90(image[:,:,0])[:,:,None], np.rot90(image[:,:,1])[:,:,None], np.rot90(image[:,:,2])[:,:,None]), axis=2)

    for idx,line in enumerate(label):
        coords = line['coords'].copy()
        new_coords = np.zeros_like(coords)
        new_coords[0,:] = 1.0 - coords[1,:]
        new_coords[1,:] = coords[0,:]
        label[idx]['coords'] = new_coords
    
    return new_image, label


def mosaic(images, labels):
    cnt = len(images)

    ratios = [image.shape[0] / image.shape[1] for image in images]
    height = []
    position = []
    target_height = np.random.randint(1500, 3000)
    random = np.random.random(10)

    if cnt == 2:
        rotate = ratios[0] < 1.0
        img_height = int(((random[1] * 0.2) + 0.8) * target_height)
        img_width_1 = int((img_height * ratios[0]) if rotate else (img_height / ratios[0]))
        position.append(((
                int((target_height - img_height) * random[2]),
                0
            ), rotate))
        height.append(img_height)

        rotate = ratios[1] < 1.0
        img_height = int(((random[3] * 0.2) + 0.8) * target_height)
        img_width_2 = int((img_height * ratios[1]) if rotate else (img_height / ratios[1]))
        position.append(((
                int((target_height - img_height) * random[4]),
                int(img_width_1 + 2)
            ), rotate))
        height.append(img_height)

        target_width = position[1][0][1] + img_width_2 + 2
    elif cnt == 3:
        rotate = ratios[0] < 1.0
        img_height_1 = int(((random[1] * 0.2) + 0.8) * target_height)
        img_width_1 = int((img_height_1 * ratios[0]) if rotate else (img_height_1 / ratios[0]))
        position.append(((
                int((target_height - img_height_1) * random[2]),
                0
            ), rotate))
        height.append(img_height_1)

        rotate = ratios[1] > 1.0
        img_height_2 = int(((random[3] * 0.2) + 0.4) * target_height)
        img_width_2 = int((img_height_2 * ratios[1]) if rotate else (img_height_2 / ratios[1]))
        position.append(((
                int((target_height - img_height_2) * random[4] / 20),
                int(img_width_1 + 2)
            ), rotate))
        height.append(img_height_2)

        gap = (position[1][0][0] + img_height_2)

        rotate = ratios[2] > 1.0
        img_height_3 = int(((random[5] * 0.1) + 0.9) * (target_height - gap))
        img_width_3 = int((img_height_3 * ratios[2]) if rotate else (img_height_3 / ratios[2]))
        position.append(((
                int((target_height - gap - img_height_3) * random[6] + gap),
                int(img_width_1 + 2)
            ), rotate))
        height.append(img_height_3)

        target_width = int(img_width_1 + 2) + max(img_width_2, img_width_3) + 2
    
    final_image = np.zeros((target_height, target_width, 3), dtype=np.int8) + 255
    final_label = []

    for i in range(cnt):
        pos, rot = position[i]
        h = height[i]
        w = int((h * ratios[i]) if rot else (h / ratios[i]))
        img, lbl = images[i], labels[i]

        if rot:
            img, lbl = rotate90(img, lbl)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        final_image[pos[0]:pos[0] + h, pos[1]:pos[1] + w, :] = img[:,:,:]
        
        for idx,line in enumerate(lbl):
            coords = line['coords'].copy() * np.array([[h], [w]]) + np.array([[pos[0]], [pos[1]]])
            lbl[idx]['coords'] = coords / np.array([[target_height], [target_width]])
        final_label += lbl

    if random[0] < 0.5:
        final_image, final_label = rotate90(final_image, final_label)

    return final_image, final_label


def flip(image, label):
    img = image.copy()
    random = np.random.random()
    do_vert = random > (1/3)
    do_hor = random < (2/3)

    for idx,item in enumerate(label):
        coords = item['coords'].copy()
        if do_vert:
            coords[0,:] = 1.0 - coords[0,:]
        if do_hor:
            coords[1,:] = 1.0 - coords[1,:]
        label[idx]['coords'] = coords
    
    if do_vert:
        img[:,:,:] = img[::-1,:,:]
    if do_hor:
        img[:,:,:] = img[:,::-1,:]
    
    return img, label


def skew(image, label):
    _lambda = np.random.random() * 0.3
    is_vert = np.random.random() > 0.5
    x_lambda = _lambda if is_vert else 0
    y_lambda = 0 if is_vert else _lambda
    shear_transformation = np.array([
        [1, y_lambda, 0],
        [x_lambda, 1, 0],
        [0, 0, 1]
    ])

    height, width, colors = image.shape

    sheared_image = ndimage.affine_transform(image, shear_transformation, offset=(int(-width * y_lambda), int(-height * x_lambda), 0), output_shape=(int(height + width * y_lambda), int(width + height * x_lambda), colors), cval=255)

    new_shape = np.array((int(height + width * y_lambda), int(width + height * x_lambda)))[:,None]

    for idx,line in enumerate(label):
        coords = line['coords'].copy() * np.array([[height], [width]])
        coords = np.dot(shear_transformation[:2,:2] * np.array([[1, -1], [-1, 1]]), coords) + np.array([[int(width * y_lambda)], [int(height * x_lambda)]])
        
        label[idx]['coords'] = (coords) / new_shape


    return sheared_image, label


def main(args):
    img_dir = args.data_directory
    lbl_dir = args.label_directory
    count = args.count

    data = [file.name.split('.')[0] for file in os.scandir(img_dir) if file.is_file()]
    weights = np.array([5.5 if datum.startswith('O') else 1 for datum in data])
    weights = weights / weights.sum()

    operation = ['c', 'f', 's']
    len_operation = len(operation)
    operation_map = {
        'c': color_space_augmentation,
        'f': flip,
        's': rotate90,
    }

    for i in range(count):
        index = i + 1

        pick = 1
        prefix = 'A'
        do_mosaic = np.random.random() > 0.5
        if do_mosaic:
            pick = np.random.randint(2, 4)
            prefix = 'M'
        
        data_choice = np.random.choice(data, pick, p=weights)
        images = load_image(img_dir, data_choice)
        labels = load_label(lbl_dir, data_choice)
        transformed_images = []
        transformed_labels = []
        for image, label in zip(images, labels):
            operation_choice = operation.copy()
            np.random.shuffle(operation_choice)
            operation_choice = operation_choice[:np.random.randint(len_operation) + 1]

            for c in operation_choice:
                image, label = operation_map[c](image, label)

            transformed_images.append(image)
            transformed_labels.append(label)
        
        if do_mosaic:
            result_image, result_label = mosaic(transformed_images, transformed_labels)
        else:
            result_image, result_label = transformed_images[0], transformed_labels[0]

        cv2.imwrite(f'{img_dir}/{prefix}{index}.png', result_image)
        write_label(f'{lbl_dir}/{prefix}{index}.txt', result_label)


def write_label(path, label):
    with open(path, 'w') as f:
        for idx,line in enumerate(label):
            if idx > 0:
                f.write('\n')
            f.write(f'{line["label"]}')
            coords = np.zeros(line["coords"].shape[1] * 2, dtype=float)
            coords[0::2] = line["coords"][0,:]
            coords[1::2] = line["coords"][1,:]
            for coord in coords:
                f.write(f' {coord}')


def load_image(root, data):
    images = []

    for datum in data:
        images.append(openRgb(f'{root}/{datum}.png'))
    return images


def load_label(root, data):
    labels = []

    for datum in data:
        with open(f'{root}/{datum}.txt') as f:
            lines = f.read().split('\n')
        
        label = []
        for line in lines:
            chars = line.split(' ')
            coords = np.array(list(map(float, chars[1:])))
            coords = np.vstack((coords[0::2], coords[1::2]))

            label.append({
                'label': chars[0],
                'coords': coords
            })

        labels.append(label)

    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = 'Dataset Augmentation',
        description = 'This program augment and create new dataset. Only for internal usage.',
        epilog = 'Created by Christian Budhi Sabdana aka MaclaurinSeries (GitHub)'
    )
    parser.add_argument('-d', '--data-dir',
                        type=str,
                        default='.\\input',
                        help='data directory',
                        dest='data_directory')
    parser.add_argument('-l', '--label-dir',
                        type=str,
                        default='.\\input',
                        help='label directory',
                        dest='label_directory')
    parser.add_argument('-c', '--count',
                        type=int,
                        default=3,
                        help='augmentation data count',
                        dest='count')
    
    main(parser.parse_args())