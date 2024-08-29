import os
import zipfile
import gdown
from PIL import Image
from torch.utils.data import Dataset
import re
import random

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CELEBA_DIR = os.path.join(CUR_DIR, './data/celeba')


def download_dataset(root_dir=CELEBA_DIR):
    """Download dataset, return dataset_folder"""

    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    dataset_folder = f'{root_dir}/img_align_celeba/'
    dataset_folder = os.path.abspath(dataset_folder)

    if not os.path.isdir(dataset_folder):
        download_url = 'https://drive.google.com/uc?export=download&id=1EehoVhQuEd-2J1EHHU1EwwmkCWYJ77qd'
        download_path = f'{root_dir}/img_align_celeba.zip'
        gdown.download(download_url, download_path, quiet=False)

        with zipfile.ZipFile(download_path, 'r') as ziphandler:
            ziphandler.extractall(root_dir)


# Create a Source Celeba Dataset class
class SourceDataset(Dataset):
    def __init__(self, root_dir=CELEBA_DIR, transform=None, selected_attrs=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        # ___________Collect images and labels__________
        self.header = selected_attrs
        self.transform = transform

        self.dataset_folder = root_dir + '/img_align_celeba/'
        self.filenames, self.targets = self._make_source_dataset(root_dir)

    def _make_source_dataset(self, root_dir):
        filenames, targets = [], []
        attr_indices = []

        with open(f'{root_dir}/list_attr_celeba.txt') as f:
            for i, line in enumerate(f.readlines()):
                line = re.sub(' *\n', '', line)
                line = re.split(' +', line)
                if i == 0:
                    for j, attr in enumerate(line):
                        if attr in self.header:
                            attr_indices.append(j)
                else:
                    filename = line[0]
                    for index, j in enumerate(attr_indices):
                        value = line[j+1]
                        if int(value) == 1:
                            filenames.append(filename)
                            targets.append(index)
        return filenames, targets

    def __getitem__(self, idx):
        img_path = self.dataset_folder+self.filenames[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.targets[idx]  # [0...num_domains-1]

        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.filenames)


# Create a Reference Celeba Dataset class
class ReferenceDataset(Dataset):
    def __init__(self, root_dir=CELEBA_DIR, transform=None, selected_attrs=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        # ___________Collect images and labels__________
        self.header = selected_attrs
        self.transform = transform

        self.dataset_folder = root_dir + '/img_align_celeba/'
        self.samples, self.targets = self._make_ref_dataset(root_dir)

    def _make_ref_dataset(self, root_dir):
        # ______Parse labels______
        domain_to_images = [[] for _ in range(len(self.header))]
        attr_indices = []

        with open(f'{root_dir}/list_attr_celeba.txt') as f:
            for i, line in enumerate(f.readlines()):
                line = re.sub(' *\n', '', line)
                line = re.split(' +', line)
                if i == 0:
                    for j, attr in enumerate(line):
                        if attr in self.header:
                            attr_indices.append(j)
                else:
                    filename = line[0]
                    for index, j in enumerate(attr_indices):
                        value = line[j+1]
                        if int(value) == 1:
                            domain_to_images[index].append(filename)

        # ____Construct dataset____
        fnames, fnames2, labels = [], [], []

        for index, cls_fnames in enumerate(domain_to_images):
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [index] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(self.dataset_folder+fname).convert('RGB')
        img2 = Image.open(self.dataset_folder+fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)
