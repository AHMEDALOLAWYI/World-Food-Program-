import torch
from torchvision import models
from torch import nn


class ContentLoss(nn.Module):
    def __init__(self, until, distance='l2'):
        super().__init__()

        self.lid = until
        self.vgg = models.vgg19(pretrained=True).features

        # No need of gradients
        for module in self.vgg.modules():
            module.requires_grad = False

        self.vgg = nn.Sequential(*list(self.vgg.children())[:until+1])
        if distance == 'l1':
            self.loss = nn.L1Loss()
        elif distance == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()

    def forward(self, reconstructed, reference):
        rec_feats = self.vgg(reconstructed)
        ref_feats = self.vgg(reference)

        b, c, h, w = ref_feats.shape
        loss_val = self.loss(rec_feats, ref_feats)
        return loss_val / (h * w)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    @staticmethod
    def get_labels(predictions, is_real):
        if is_real:
            return torch.ones_like(predictions)
        else:
            return torch.zeros_like(predictions)

    def forward(self, predictions, is_real):
        ground_truth = AdversarialLoss.get_labels(predictions, is_real)
        return self.loss(predictions, ground_truth)
