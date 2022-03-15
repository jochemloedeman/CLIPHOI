import torchvision.models as models
import torch.nn as nn


class HICOResNet(nn.Module):

    def __init__(self, pretrained=False):
        super(HICOResNet, self).__init__()
        self.model = models.resnet18(pretrained)
        self.model.fc = nn.Linear(512, 520)

        for params in self.model.parameters():
            params.requires_grad = False
        for params in self.model.fc.parameters():
            params.requires_grad = True
        # for params in self.model.layer4.parameters():
        #     params.requires_grad = True

    def forward(self, x):
        logits = self.model(x)
        return logits


if __name__ == '__main__':
    hico_resnet = HICOResNet(pretrained=False)
