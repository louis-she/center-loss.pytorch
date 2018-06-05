import torch
from torch import nn
import torch.nn.functional as F

from .base import FaceModel
from device import device


class MNISTModel(FaceModel):

    IMAGE_SHAPE = (28, 28)

    def __init__(self, num_classes):
        super().__init__(10, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(1, 64, 3, 2, padding=1) # 14 x 14 x 8
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, 2, padding=1) # 7 x 7 x 16
        self.bn2 = nn.BatchNorm2d(128)

        self.pooling = nn.AvgPool2d(7) # 1 x 1 x 16
        self.feature = nn.Linear(128, 2)

        self.fc = nn.Linear(2, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pooling(x)
        x = x.view(x.size(0), -1)

        feature = self.feature(x)
        logits = self.fc(feature)

        feature_normed = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return logits, feature_normed


class MNISTExample(FaceModel):

    FEATURE_DIM = 2

    def __init__(self, num_classes):
        super().__init__(10, 2)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 2)
        self.fc2 = nn.Linear(2, 10)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()

    def forward(self, x):
        x = self.relu1(F.max_pool2d(self.conv1(x), 2))
        x = self.relu2(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        feature = x = self.relu3(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), feature
