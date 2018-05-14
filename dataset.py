import os
import tarfile
import random
from math import ceil
from tqdm import tqdm

import requests
from torch.utils import data

def download_dataset(dir):
    url = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
    download_path = os.path.join(dir, url.split('/')[-1])
    if os.path.isfile(download_path):
        print('File {} already downloaded'.format(download_path))
        return download_path
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024

    with open(download_path, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                total=ceil(total_size//block_size),
                unit='KB', unit_scale=True):
            f.write(data)
    return download_path

def create_image_generator(dataroot, download=True, shuffle=True):
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

    if shuffle:
        names = random.shuffle(names)

    def image_generator(ratio):
        slice_at = len(names) * ratio
        ret = names[:slice_at]
        names = names[slice_at:]
        return ret

    return image_generator

class Dataset(data.Dataset):

    def __init__(self, dataroot, names, transform=None, target_transform=None,
                 is_train=True, split=0.1, loader=None):
        self.dataroot = dataroot
        self.names = names
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        self.split = split
        self.loader = loader
        self.image_names = []
        self.labels = []

        for name in self.names:
            for image_name in os.listdir(os.path.join(self.dataroot, name)):
                self.image_names.append(image_name)
                self.labels.append(name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.image_names[index], self.labels[index])