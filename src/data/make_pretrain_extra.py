from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    root = '/home/external-rosia/rosia/reflection-connection/data/processed/pretrain/'
    extra = '/home/external-rosia/rosia/reflection-connection/data/processed/pretrain/extra'
    dataset = ImageNet(split=split, root=root, extra=extra)
    dataset.dump_extra()
