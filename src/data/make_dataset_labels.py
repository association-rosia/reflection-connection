import src.utils as utils
import os


def main():
    config = utils.get_config()
    folders_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'train')
    folders = os.listdir(folders_path)
    save_path = os.path.join(config['path']['data'], 'processed', 'pretrain', 'labels.txt')

    with open(save_path, 'w') as f:
        for folder in folders:
            f.write(f'{folder}, seismic slice')
            f.write('\n')


if __name__ == '__main__':
    main()
