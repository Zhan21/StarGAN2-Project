from munch import Munch
from dataset import SourceDataset, ReferenceDataset, download_dataset

import numpy as np
import random
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms


def build_dataset(args):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size,
                                     scale=[0.8, 1.0],
                                     ratio=[0.9, 1.1]),
        transforms.Resize([args.img_size, args.img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    selected_attrs = ['Attractive', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
                      'Double_Chin', 'Male', 'Pale_Skin', 'Smiling', 'Wearing_Hat', 'Young']

    dataset = SourceDataset(transform=transform, selected_attrs=selected_attrs)
    dataset_ref = ReferenceDataset(
        transform=transform, selected_attrs=selected_attrs)

    return dataset, dataset_ref


def get_train_loader(dataset, args):
    class_counts = np.bincount(dataset.targets)
    class_weights = 1. / class_counts
    weights = class_weights[dataset.targets]
    sampler = WeightedRandomSampler(weights, len(weights))

    return DataLoader(dataset=dataset,
                      batch_size=args.batch_size,
                      sampler=sampler,
                      num_workers=args.num_workers,
                      pin_memory=True,
                      drop_last=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, device=None):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = device
        self.iter = iter(self.loader)
        self.iter_ref = iter(self.loader_ref)

    def __next__(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)

        try:
            x_ref, x_ref2, y_ref = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x_ref, x_ref2, y_ref = next(self.iter_ref)

        z_trg = torch.randn(x.size(0), self.latent_dim)
        z_trg2 = torch.randn(x.size(0), self.latent_dim)

        inputs = Munch(x_src=x, y_src=y,
                       x_ref=x_ref, x_ref2=x_ref2, y_ref=y_ref,
                       z_trg=z_trg, z_trg2=z_trg2)

        return Munch({k: v.to(self.device) for k, v in inputs.items()})
