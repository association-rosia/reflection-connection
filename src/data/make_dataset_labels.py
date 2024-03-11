import src.utils as utils
import os


def main():
    config = utils.get_config()
    folders_train_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'train')
    folders_train = os.listdir(folders_train_path)
    save_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'labels.txt')

    with open(save_path, 'w') as f:
        for i, folder in enumerate(folders_train):
            f.write(f'{folder}, seismic{i}')
            f.write('\n')


if __name__ == '__main__':
    main()
