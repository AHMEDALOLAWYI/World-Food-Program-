import torch
from torch import nn
import torch.nn.functional as F


class VATLoss(nn.Module):
    def __init__(self, ip, epsilon, xi):
        """
        The parameters used for initializing the module are the same as given
        in the paper https://arxiv.org/abs/1507.00677
        :param ip: Number of iterations for Algorithm 1
        :param epsilon: Scaling parameter for the unit vector
        :param xi: Scaling parameter for adversarial gradient
        """
        super().__init__()

        self.ip = ip
        self.epsilon = epsilon
        self.xi = xi

    def forward(self, unlab_xs, model, logits=False):

        # If our model gives logits as the output instead of
        # LogSoftmax output, we will perform the the operation here

        # We don't need gradients to be calculated because these
        # predictions would act like targets
        ys = None
        with torch.no_grad():
            ys = model(unlab_xs)

            if logits:
                ys = F.softmax(ys, dim=1)

        # Generate a random vector, convert it to a unit vector
        d = torch.rand(unlab_xs.shape).sub(0.5).to(unlab_xs.device)
        d = VATLoss.normalize(d)

        for _ in range(self.ip):
            d.requires_grad = True
            y_adv = model(unlab_xs + d * self.xi)
            if logits:
                y_adv = F.log_softmax(y_adv, dim=1)
            divergence = F.kl_div(y_adv, ys, reduction='batchmean')
            divergence.backward()

            # Use normalized gradients (that would just give us the
            # direction of adversarial perturbation)
            d = VATLoss.normalize(d.grad)

            # Set model's gradients back to zero
            model.zero_grad()

        # Having got the adversarial direction, calculate the VATLoss
        r_adv = self.epsilon * d
        y_adv = model(unlab_xs + r_adv)
        if logits:
            y_adv = F.log_softmax(y_adv, dim=1)
        divergence = F.kl_div(y_adv, ys, reduction='batchmean')
        return divergence





    @staticmethod
    def normalize(tensor):
        t_reshaped = tensor.view(tensor.size(0), -1, 1, 1)
        tensor /= torch.norm(t_reshaped, dim=1, keepdim=True) + 1e-10
        return tensor
