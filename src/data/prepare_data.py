import os
import shutil
import torch
from src import utils
from tqdm.autonotebook import tqdm
from torchvision.datasets import ImageFolder
import src.data.transforms as dT


def merge_train_val_data():
    config = utils.get_config()
    folder1 = os.path.join(config['path']['data']['raw']['train'], 'train')
    folder2 = os.path.join(config['path']['data']['raw']['train'], 'test')
    destination = config['path']['data']['raw']['train']
    merge_folders(folder1, folder2, destination)


def merge_folders(folder1, folder2, destination):
    os.makedirs(destination, exist_ok=True)
    shutil.copytree(folder1, destination, dirs_exist_ok=True)
    shutil.copytree(folder2, destination, dirs_exist_ok=True)
    shutil.rmtree(folder1)
    shutil.rmtree(folder2)


def compute_image_mean_std(contrast=25):
    config = utils.get_config()
    # TODO: Faire un processor sans croping
    wandb_config = utils.load_config('dinov2.yml')
    processor = dT.make_eval_processor(config, wandb_config)
    dataset = ImageFolder(config['path']['data']['raw']['train'], transform=processor)
    # All images have not the same shape
    sum = torch.tensor(0.)
    count = torch.tensor(0.)
    print('\nCompute mean:')
    for img, _ in tqdm(dataset):
        sum += torch.sum(img)
        count += img.nelement()

    mean = sum / count

    sum = torch.tensor(0.)
    print('\nCompute std:')
    for img, _ in tqdm(dataset):
        sum += torch.sum((img - mean) ** 2)

    std = torch.sqrt(sum / count)
    
    print('mean', mean)
    print('std', std)


if __name__ == "__main__":
    # merge_train_val_data()
    compute_image_mean_std()