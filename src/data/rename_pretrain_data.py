import src.utils as utils
from tqdm.auto import tqdm
import os


def main():
    config = utils.get_config()
    folders_path = os.path.join(config['path']['data'], 'processed', 'pretrain')

    for split in tqdm(['train', 'val']):
        folders = os.listdir(os.path.join(folders_path, split))

        for folder in tqdm(folders):
            new_folder = folder.replace('_', '').replace('.', '')
            images = os.listdir(os.path.join(folders_path, split, folder))

            folder_path = os.path.join(folders_path, split, folder)
            new_folder_path = os.path.join(folders_path, split, new_folder)
            os.rename(folder_path, new_folder_path)

            for i, image in tqdm(enumerate(images)):
                new_image = f'{new_folder}_{i}'
                image_path = os.path.join(folders_path, split, new_folder, image)
                new_image_path = os.path.join(folders_path, split, new_folder, new_image)
                os.rename(image_path, new_image_path)


if __name__ == '__main__':
    main()
