import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(1, 40, kernel_size=3, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.weight_norm(nn.Conv2d(40, 40, kernel_size=3, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.weight_norm(nn.Conv2d(40, 40, kernel_size=3, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear(46240,256)
                )
                ])

    def forward(self, x):
        '''
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        for module in self.discriminator:
            x = module(x)
        return x

class SFDiscriminator(nn.Module):
    def __init__(self):
        super(SFDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [Discriminator() for _ in range(3)]
        )

    def forward(self, x):
        # x - input mel of size [B, 1, 40, 80]
        x_in = [ x[0:16, 0:1, 0:40, 0:40], x[0:16, 0:1, 0:40, 20:60], x[0:16, 0:1, 0:40, 40:80] ]
        disc_out = list()


        for disc, x_ in zip(self.discriminators, x_in):
            x = disc(x_)
            disc_out.append(x)

        return disc_out # [SF_out0, SF_out1, SF_out2]


if __name__ == '__main__':
    model = SFDiscriminator()

    x = torch.randn(16, 1, 40, 80)
    print(x.shape)

    out = model(x)
    print(len(out), "Shape of output")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
