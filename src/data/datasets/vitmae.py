from PIL import Image
from torch.utils.data import Dataset

import src.data.transforms as dT
from src import utils
from src.data import utils as d_utils


class RefConViTMAEDataset(Dataset):

    def __init__(self, wandb_config, images_path: list, processor: dT.RefConfProcessor):
        self.wandb_config = wandb_config
        self.images_path = images_path
        self.processor = processor

    def __len__(self):
        return len(self.images_path)

    def _load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return self.processor.preprocess_image(img)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        item = self._load_image(image_path)

        return item


def make_pretrain_dataset(config, wandb_config, set):
    images_path = d_utils.get_images_path(config, set)
    processor = dT.make_pretraining_processor(config, wandb_config)

    return RefConViTMAEDataset(wandb_config, images_path, processor)


def _debug():
    from tqdm.autonotebook import tqdm

    config = utils.get_config()
    wandb_config = utils.load_config('pretraining/vitmae.yml')
    dataset = make_pretrain_dataset(config, wandb_config, set='val')

    for inputs in tqdm(dataset):
        pass

    pass


if __name__ == '__main__':
    _debug()
