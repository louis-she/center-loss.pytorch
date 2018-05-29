import os
import argparse

import torch
from torchvision import transforms

from dataset import Dataset, create_datasets
from loss import compute_center_loss, get_center_delta
from model import FaceModel
from device import device
from trainer import Trainer

parser = argparse.ArgumentParser(description='center loss example')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.001')
parser.add_argument('--resume', type=str, help='model path to the resume training', default=False)

if __name__ == '__main__':
    args = parser.parse_args()

    home = os.path.expanduser("~")
    dataset_root = os.path.join(home, 'datasets', 'lfw')
    training_set, validation_set, num_classes = create_datasets(dataset_root)

    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((96,128)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    validation_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((96,128)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    training_dataset = Dataset(training_set, train_transforms)
    validation_dataset = Dataset(validation_set, validation_transforms)

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, num_workers=6, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, num_workers=6, shuffle=False)

    model = FaceModel(num_classes).to(device)
    model.load_state_dict(torch.load( './resnet18.pth' ), strict=False)

    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
    optimizer = torch.optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=args.lr, momentum=0.9)

    trainer = Trainer(optimizer, model, training_dataloader, validation_dataloader, max_epoch=args.epochs, resume=args.resume)
    trainer.train()