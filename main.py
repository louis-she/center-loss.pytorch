import os

import torch
from torchvision import transforms

from dataset import Dataset, create_image_generator
from loss import compute_center_loss, get_center_delta
from model import FaceModel
from device import device

if __name__ == '__main__':
    #TODO: add arg control
    home = os.path.expanduser("~")
    dataset_root = os.path.join(home, 'datasets', 'flw')
    data_generator = create_image_generator(dataset_root)

    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((96,128)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    training_dataset = Dataset(dataset_root, data_generator(0.8), train_transforms)
    validation_dataset = Dataset(dataset_root, data_generator(1))

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=16)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=16)

    model = FaceModel(training_dataset.num_classes).to(device)

    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
    optimizer = torch.optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=0.001, momentum=0.9)

    lamda = 0.003
    alpha = 0.5

    model._buffers['centers'] = torch.rand(training_dataset.num_classes, 512).to(device)  - 0.5 * 2

    for images, targets, names in training_dataloader:
        targets = torch.tensor(targets).to(device)
        images = images.to(device)

        logits, features = model(images)
        cross_entropy_loss = torch.nn.functional.cross_entropy(logits, targets)
        center_loss = compute_center_loss(features, model._buffers['centers'], targets, lamda)
        loss = cross_entropy_loss + center_loss
        print("cross entropy loss: {} - center loss: {} - total loss: {}".format(cross_entropy_loss, center_loss, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update center
        center_deltas = get_center_delta(features, model._buffers['centers'], targets, alpha)
        model._buffers['centers'] = model._buffers['centers'] - center_deltas