import torch
from torch import nn
import model_utils
from collections import OrderedDict

DEVICE = 'cuda'


class ResBlock(nn.Module):
    def __init__(self, inf, outf):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inf)
        self.conv1 = nn.Conv2d(inf, outf, 3, 1, 1, bias=True)
        self.prelu1 = nn.PReLU()
        self.bn2 = nn.BatchNorm2d(outf)
        self.conv2 = nn.Conv2d(outf, outf, 3, 1, 1, bias=True)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        z = self.conv1(self.prelu1(self.bn1(x)))
        z = self.conv2(self.prelu2(self.bn2(z)))
        z = x + z
        return z


class GeneratorBNFirst(nn.Module):
    def __init__(self, in_chan, out_chan, n_planes=64, n_blocks=16, upscale=4):
        assert upscale in [2, 4], 'Only supports scaling the image by either 2 or 4'
        super().__init__()
        self.upscale = upscale
        self.first_conv = nn.Conv2d(in_chan, n_planes, 9, 1, 4)
        model_utils.w_init([self.first_conv])
        self.first_prelu = nn.PReLU()

        # Create the intermediate residual blocks
        # (B residual blocks in the paper)
        blocks = list()
        for _ in range(n_blocks):
            blk = ResBlock(n_planes, n_planes)
            model_utils.w_init([blk])
            model_utils.g_init([blk])
            blocks.append(blk)

        self.B = nn.Sequential(*blocks)

        self.mid_conv = nn.Conv2d(n_planes, n_planes, 3, 1, 1)
        self.mid_bn = nn.BatchNorm2d(n_planes)

        # Block for upscaling

        self.upscale_block = nn.Sequential(OrderedDict([
                                ('up_conv1', nn.Conv2d(n_planes, 4 * n_planes, 3, 1, 1)),
                                ('pix_shuffle1', nn.PixelShuffle(2)),
                                ('up_prelu1', nn.PReLU())
                                ]))
        if self.upscale == 4:
            self.upscale_block.add_module('up_conv2', nn.Conv2d(n_planes, 4 * n_planes, 3, 1, 1))
            self.upscale_block.add_module('pixel_shuffle2', nn.PixelShuffle(2))
            self.upscale_block.add_module('up_prelu2', nn.PReLU())

        model_utils.w_init([self.upscale_block])

        self.end_conv = nn.Conv2d(n_planes, out_chan, 9, 1, 4)
        model_utils.w_init([self.end_conv])

    def forward(self, x):
        z = self.first_prelu(self.first_conv(x))
        trunk = self.B(z)
        trunk = self.mid_conv(self.mid_bn(trunk))
        # The giant skip connection
        trunk = trunk + z

        # Upscaling
        trunk = self.upscale_block(trunk)
        trunk = self.end_conv(trunk)
        return trunk


if __name__ == '__main__':
    # Test architecture
    gen = GeneratorBNFirst(3, 3, upscale=4)
    gen.to(DEVICE)
    # print(gen)
    img = torch.rand(1, 3, 256, 256).to(DEVICE)
    out = gen(img)
    print(out.shape)
