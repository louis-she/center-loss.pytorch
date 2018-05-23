import os
import tarfile
import random
from math import ceil, floor
from tqdm import tqdm

import requests
from torch.utils import data

from utils import image_loader

def download_dataset(dir):
    url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
    download_path = os.path.join(dir, url.split('/')[-1])
    if os.path.isfile(download_path):
        print('File {} already downloaded'.format(download_path))
        return download_path
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 * 1024

    with open(download_path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                total=ceil(total_size//block_size),
                unit='KB', unit_scale=True):
            f.write(data)
    return download_path

def create_datasets(dataroot, download=True, train_val_split=0.9):
    if not os.path.isdir(dataroot):
        if download is False:
            raise RuntimeError('Dataroot {} is not a directory'
                .format(dataroot))
        else:
            print('Download dataset to {}'.format(dataroot))
            os.mkdir(dataroot)
            tarball = download_dataset(os.path.dirname(dataroot))

            print('Dataset downloaded to {}'.format(dataroot))
            print('Extract dataset to {}'.format(dataroot))

            def members(t, skipped_lenth):
                for member in t.getmembers():
                    member.path = member.path[l:]
                    yield member

            with tarfile.open(tarball, 'r') as t:
                l = len(tarball.split('/')[-1].split('.')[0]) + 1
                t.extractall(dataroot, members=members(t, l))

    names = os.listdir(dataroot)
    if len(names) == 0:
        raise RuntimeError('Empty dataset')

    training_set = []
    validation_set = []
    for klass, name in enumerate(names):
        def add_class(image):
            image_path = os.path.join(dataroot, name, image)
            return (image_path, klass, name)

        images_of_person = os.listdir(os.path.join(dataroot, name))
        total = len(images_of_person)

        training_set += map(add_class, images_of_person[ :ceil(total * train_val_split) ])
        validation_set += map(add_class, images_of_person[ floor(total * train_val_split): ])

    return training_set, validation_set, len(names)

class Dataset(data.Dataset):

    def __init__(self, datasets, transform=None, target_transform=None):
        self.datasets = datasets
        self.num_classes = len(datasets)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        image = image_loader(self.datasets[index][0])
        if self.transform:
            image = self.transform(image)
        return (image, self.datasets[index][1], self.datasets[index][2])