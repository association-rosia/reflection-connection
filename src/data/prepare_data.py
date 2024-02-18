import os
import shutil
from src import utils

def merge_folders(folder1, folder2, destination):
    os.makedirs(destination, exist_ok=True)
    shutil.copytree(folder1, destination, dirs_exist_ok=True)
    shutil.copytree(folder2, destination, dirs_exist_ok=True)
    shutil.rmtree(folder1)
    shutil.rmtree(folder2)


if __name__ == "__main__":
    config = utils.get_config()
    folder1 = os.path.join(config['path']['data']['raw']['train'], 'train')
    folder2 = os.path.join(config['path']['data']['raw']['train'], 'test')
    destination = config['path']['data']['raw']['train']

    merge_folders(folder1, folder2, destination)
