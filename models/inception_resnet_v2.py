import pretrainedmodels
import torch
import torch.nn as nn

from .base import FaceModel

class InceptionResnetV2(FaceModel):

    IMAGE_SHAPE = (160, 160)
    FEATURE_DIM = 512

    def __init__(self, num_classes):
        super().__init__(num_classes, self.FEATURE_DIM)
        self.base = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        self.pool = nn.AvgPool2d(3, count_include_pad=False)

        self.extract_feature = nn.Linear(1536*3*3, self.feature_dim)

        self.num_classes = num_classes
        if self.num_classes:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.base.features(x)
        x = x.view(x.size(0), -1)

        feature = self.extract_feature(x)
        logits = self.classifier(self.relu(feature)) if self.num_classes else None
        feature_normed = feature.div(
            torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))

        return logits, feature, feature_normed