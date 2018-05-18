import os

import torch

from dataset import Dataset, create_image_generator
from loss import compute_center_loss
from model import FaceModel

if __name__ == '__main__':
    #TODO: add arg control
    home = os.path.expanduser("~")
    dataset_root = os.path.join(home, 'datasets', 'flw')
    data_generator = create_image_generator(dataset_root)

    training_dataset = Dataset(dataset_root, data_generator(0.8))
    validation_dataset = Dataset(dataset_root, data_generator(1))

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=16)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=16)

    image, target, name = training_dataset[0]
    torch.tensor(image)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

    model = FaceModel(training_dataset.num_classes)
    logits, features = model(image)
    print(logits, features)