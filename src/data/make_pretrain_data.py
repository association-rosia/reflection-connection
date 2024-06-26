import os
import random
from glob import glob

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import src.utils as utils


def main():
    # check_data()
    config = utils.get_config()
    init_folders(config)
    extract_tiles_from_volumes(config)
    count_images(config)


def get_values_counts(config):
    shapes = []
    data_train_path = os.path.join(config['path']['data'], 'raw', 'train')
    data_train_glob = os.path.join(data_train_path, '**/*.png')

    for image_path in glob(data_train_glob, recursive=True):
        image = Image.open(image_path).convert('L')
        shapes.append(np.array(image).shape)

    values, counts = np.unique(shapes, return_counts=True, axis=0)

    return values, counts


def draw_random_shape(values, counts):
    p = counts / counts.sum()
    index = np.random.choice(len(values), p=p)
    random_shape = values[index]

    return random_shape


def normalize_pretrain_slice(slice):
    slice = cv2.normalize(slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)

    return slice


def adjust_coordinates(x0, y0, h, w, max_h, max_w):
    x1 = x0 + h
    y1 = y0 + w

    if x1 > max_h:
        x0 = max_h - h
        x1 = max_h

    if y1 > max_w:
        y0 = max_w - w
        y1 = max_w

    return x0, x1, y0, y1


def get_tiles_coords(values, counts, num_tiles=4, max_h=1259, max_w=300):
    tiles_coords = []

    for x0 in range(0, max_h, 126):
        for y0 in range(0, max_w, 100):
            h, w = draw_random_shape(values, counts)
            x0, x1, y0, y1 = adjust_coordinates(x0, y0, h, w, max_h, max_w)
            tiles_coords.append((x0, x1, y0, y1))

    tiles_coords = random.sample(tiles_coords, k=num_tiles)

    return tiles_coords


def extract_tiles_from_slice(slice, save_volume_path, values, counts, volume_name, image_idx):
    tiles_coords = get_tiles_coords(values, counts)

    for x0, x1, y0, y1 in tiles_coords:
        tile = slice[x0:x1, y0:y1]
        tile = normalize_pretrain_slice(tile)
        image = Image.fromarray(tile).convert('L')
        save_image_path = os.path.join(save_volume_path, f'{volume_name}-{image_idx}.png')
        image.save(save_image_path)
        image_idx += 1

    return image_idx


def extract_tiles_from_volumes(config):
    values, counts = get_values_counts(config)
    data_pretrain_path = os.path.join(config['path']['data'], 'raw', 'pretrain')
    save_pretrain_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
    data_pretrain_glob = os.path.join(data_pretrain_path, '**/*.npy')

    for volume_path in tqdm(glob(data_pretrain_glob, recursive=True)):
        volume = np.load(volume_path)

        volume_name = volume_path.split('/')[-1].replace('.npy', '')
        volume_name = volume_name.replace('.', '').replace('_', '')
        save_volume_path = os.path.join(save_pretrain_path, volume_name)
        os.makedirs(save_volume_path, exist_ok=True)

        image_idx = 0
        for slice_idx in range(len(volume)):
            slice = volume[slice_idx, :, :].T
            image_idx = extract_tiles_from_slice(slice, save_volume_path, values, counts, volume_name, image_idx)
            slice = volume[:, slice_idx, :].T
            image_idx = extract_tiles_from_slice(slice, save_volume_path, values, counts, volume_name, image_idx)


def init_folders(config):
    pretrain_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
    os.makedirs(pretrain_path, exist_ok=True)
    pretrain_train_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
    os.makedirs(pretrain_train_path, exist_ok=True)


def check_data():
    config = utils.get_config()
    data_pretrain_path = os.path.join(config['path']['data'], 'raw', 'pretrain')
    data_pretrain_glob = os.path.join(data_pretrain_path, '**/*.npy')

    pbar = tqdm(glob(data_pretrain_glob, recursive=True))
    for volume_path in pbar:
        try:
            _ = np.load(volume_path)
        except Exception:
            pbar.write(f'ERROR - {volume_path}')


def count_images(config):
    pretrain_train_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
    pretrain_train_glob = os.path.join(pretrain_train_path, '**/*.png')
    images = glob(pretrain_train_glob, recursive=True)
    print(f'Final number of images: {len(images)}')


if __name__ == "__main__":
    main()
