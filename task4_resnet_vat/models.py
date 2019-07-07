from torch import nn
import torch.nn.functional as F


class Resnet(nn.Module):
    def __init__(self, arch, num_classes):
        resnet = arch(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


class EfficientNet(nn.Module):
    pass
