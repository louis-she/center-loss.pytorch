import os

import torch
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from utils import download
from metrics import compute_roc
from model import FaceModel
from dataset import LFWPairedDataset
from device import device


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return pairs

def evaluate(embedings_a, embedings_b, matches, thresholds, dis_metric='euclid'):
    """
    Args:
        features: array of pair features, [ ([feature_1], [feature_2], is_matched), ... ]
        thresholds: candidates for possible thresholds
    Returns:
        threshold which has the most accuracy
    """
    # 1. compute distance
    if dis_metric == 'euclid':
        distances = torch.sum(torch.pow(embedings_a - embedings_b, 2), dim=1)

    # 2. calculate roc
    return compute_roc(distances, matches, thresholds)

if __name__ == '__main__':
    home = os.path.expanduser("~")
    dataset_dir = os.path.join(home, 'datasets', 'lfw')

    batch_size = 128
    dataroot = '/home/louis/datasets/lfw'

    pairs_path = os.path.join(dataroot, 'pairs.txt')
    if not os.path.isfile(pairs_path):
        download(dataroot, 'http://vis-www.cs.umass.edu/lfw/pairs.txt')

    transforms = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((96,128)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    dataset = LFWPairedDataset(dataroot, pairs_path, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    model = FaceModel().to(device)
    checkpoint = torch.load('./logs/models/epoch_50.pth.tar')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    embedings_a = torch.zeros(len(dataset), 512)
    embedings_b = torch.zeros(len(dataset), 512)
    matches = torch.zeros(len(dataset), dtype=torch.uint8)

    for iteration, (images_a, images_b, batched_matches) in enumerate(dataloader):
        current_batch_size = len(batched_matches)
        images_a = images_a.to(device)
        images_b = images_b.to(device)

        _, batched_embedings_a = model(images_a)
        _, batched_embedings_b = model(images_b)

        start = batch_size * iteration
        end = start + current_batch_size

        embedings_a[start:end, :] = batched_embedings_a.data
        embedings_b[start:end, :] = batched_embedings_b.data
        matches[start:end] = batched_matches.data

    thresholds = np.arange(0, 4, 0.1)
    tpr, fpr, accuracy = evaluate(embedings_a, embedings_b, matches, thresholds, 'cosine')
    print(accuracy)

    fig = plt.figure()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr)
    fig.savefig('/tmp/temp.png', dpi=fig.dpi)

