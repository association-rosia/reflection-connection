import os
import random
import shutil
from glob import glob

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import src.utils as utils


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
    return cv2.normalize(slice, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC1)


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


def get_tiles_coords(values, counts, num_tiles=15, max_h=1259, max_w=300):
    tiles_coords = []

    for x0 in range(0, max_h, 126):
        for y0 in range(0, max_w, 100):
            h, w = draw_random_shape(values, counts)
            x0, x1, y0, y1 = adjust_coordinates(x0, y0, h, w, max_h, max_w)
            tiles_coords.append((x0, x1, y0, y1))

    tiles_coords = random.choices(tiles_coords, k=num_tiles)

    return tiles_coords


def extract_tiles_from_slice(slice, save_volume_path, dim, slice_idx, values, counts):
    tiles_coords = get_tiles_coords(values, counts)
    os.makedirs(save_volume_path, exist_ok=True)

    for i, (x0, x1, y0, y1) in enumerate(tiles_coords):
        tile = slice[x0:x1, y0:y1]
        tile = normalize_pretrain_slice(tile)
        image = Image.fromarray(tile).convert('L')
        save_image_path = os.path.join(save_volume_path, f'{dim}-{slice_idx}-{i}.png')
        image.save(save_image_path)


def split_tiles(save_volume_path, volume_name):
    save_train_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'train')
    save_val_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'val', volume_name)
    os.makedirs(save_val_path, exist_ok=True)

    tile_files = os.listdir(save_volume_path)
    _, tile_files_val = train_test_split(tile_files, test_size=0.01, random_state=42)

    for tile_file in tile_files_val:
        tile_path = os.path.join(save_volume_path, tile_file)
        new_tile_path = os.path.join(save_val_path, tile_file)
        shutil.move(tile_path, new_tile_path)

    shutil.move(save_volume_path, save_train_path)


def extract_tiles_from_volumes(config):
    values, counts = get_values_counts(config)
    data_pretrain_path = os.path.join(config['path']['data'], 'raw', 'pretrain')
    save_pretrain_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
    data_pretrain_glob = os.path.join(data_pretrain_path, '**/*.npy')

    for volume_path in tqdm(glob(data_pretrain_glob, recursive=True)):
        volume = np.load(volume_path)

        volume_name = volume_path.split('/')[-1].replace('.npy', '')
        save_volume_path = os.path.join(save_pretrain_path, volume_name)

        for slice_idx in range(len(volume)):
            slice = volume[slice_idx, :, :].T
            extract_tiles_from_slice(slice, save_volume_path, 0, slice_idx, values, counts)
            slice = volume[:, slice_idx, :].T
            extract_tiles_from_slice(slice, save_volume_path, 1, slice_idx, values, counts)

        split_tiles(save_volume_path, volume_name)


def init_folders(config):
    pretrain_path = os.path.join(config['path']['data'], 'processed', 'pretrain')
    os.makedirs(pretrain_path, exist_ok=True)
    pretrain_train_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'train')
    os.makedirs(pretrain_train_path, exist_ok=True)
    pretrain_val_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'val')
    os.makedirs(pretrain_val_path, exist_ok=True)


if __name__ == "__main__":
    config = utils.get_config()
    init_folders(config)
    extract_tiles_from_volumes(config)
