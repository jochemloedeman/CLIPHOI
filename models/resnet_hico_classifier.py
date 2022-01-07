import torchvision.models as models
import torch.nn as nn


class HICOResNet(nn.Module):

    def __init__(self):
        super(HICOResNet, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=False)
        self.classifier = nn.Linear(1000, 520)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
