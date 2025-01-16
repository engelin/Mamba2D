import torch
import lightning as L

from torchvision.transforms import v2
from torch.utils.data import default_collate
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from timm.data import create_transform, Mixup

from .datasets import ImageNet1k

class MNISTDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str = "datasets",
                 fashionMNIST: bool = False,
                 batch_size: int = 32,
                 num_workers: int = 19):
        super().__init__()
        self.data_dir = data_dir
        self.fashionMNIST = fashionMNIST
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        if (self.fashionMNIST):
            MNISTset = FashionMNIST
        else:
            MNISTset = MNIST

        self.train_dataset = MNISTset(root=self.data_dir,
                                   download=True,
                                   train=True,
                                   transform=tf)

        self.val_dataset = MNISTset(root=self.data_dir,
                                   download=True,
                                   train=False,
                                   transform=tf)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)

class CIFARDataModule(L.LightningDataModule):
    def __init__(self,
                 data_dir: str = "datasets",
                 batch_size: int = 32,
                 num_workers: int = 19):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        self.train_dataset = CIFAR10(root=self.data_dir,
                                   download=True,
                                   train=True,
                                   transform=tf)

        self.val_dataset = CIFAR10(root=self.data_dir,
                                   download=True,
                                   train=False,
                                   transform=tf)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)
    
class ImageNetDataModule(L.LightningDataModule):
    def __init__(self,
                 # DataModule Params
                 data_dir: str = "datasets",
                 batch_size: int = 32,
                 num_workers: int = 4,

                 # Speed Params
                 channel_last: bool = False,
                 img_dtype: torch.dtype | None = torch.float32,
                 
                 # Augmentation Params
                 cutmix: bool = True,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.channel_last = channel_last
        self.img_dtype = img_dtype
        self.cutmix = cutmix

    def setup(self, stage: str):
        # Transforms based on common practices in DeiT etc.
        train_tf = create_transform(
            input_size=(3,224,224),
            is_training=True,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            hflip=0.5,
            vflip=0,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='random',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )

        val_tf = create_transform(
            input_size=(3,224,224),
            is_training=False,
            interpolation='bicubic',
            crop_pct=1.0,
        )

        self.cutmix_tf = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=0.1,
            num_classes=1000
        )
        
        self.train_dataset = ImageNet1k(root = self.data_dir + '/TrainingSet',
                                        split = 'train',
                                        transform=train_tf)

        self.val_dataset = ImageNet1k(root = self.data_dir + '/ValidationSet',
                                      split = 'val',
                                      transform=val_tf)

    def batch_tf_collate(self, batch):
        images, labels = default_collate(batch)

        if self.channel_last:
            images = images.to(memory_format=torch.channels_last)
        
        if self.img_dtype is not None: 
            images = images.to(dtype=self.img_dtype)
        
        return (images,labels)
    
    def cutmix_batch_tf_collate(self, batch):
        return self.cutmix_tf(*self.batch_tf_collate(batch))


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.cutmix_batch_tf_collate if self.cutmix else self.batch_tf_collate,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.batch_tf_collate,
            pin_memory=True,
            persistent_workers=False
        )
    