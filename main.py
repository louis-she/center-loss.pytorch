import os

import torch
from torchvision import transforms

from dataset import Dataset, create_image_generator
from loss import compute_center_loss, get_center_delta
from model import FaceModel
from device import device
from trainer import Trainer

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

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=128)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=128)

    model = FaceModel(training_dataset.num_classes).to(device)
    model.load_state_dict(torch.load( './resnet18.pth' ), strict=False)

    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
    optimizer = torch.optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=0.001, momentum=0.9)

    lamda = 0.003
    alpha = 0.5

    model._buffers['centers'] = torch.rand(training_dataset.num_classes, 512).to(device)  - 0.5 * 2
    max_epoch = 30

    trainer = Trainer(optimizer, model, training_dataloader, validation_dataloader)
    trainer.train()