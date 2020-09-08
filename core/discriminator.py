import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding = 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding = 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding = 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding = 1)
                #nn.Flatten(),   # add conv2d a 1 channel
                #nn.Linear(46240,256)
                )

    def forward(self, x):
        '''
        we directly predict score without last sigmoid function
        since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        # print(x.shape, "Input to Discriminator")
        return self.discriminator(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class SFDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc1 = Discriminator()
        self.disc2 = Discriminator()
        self.disc3 = Discriminator()
        self.apply(weights_init)
    def forward(self, x, start):
        results = []
        results.append(self.disc1(x[:, :, start: start + 40, 0:40]))
        results.append(self.disc2(x[:, :, start: start + 40, 20:60]))
        results.append(self.disc3(x[:, :, start: start + 40, 40:80, ]))
        return results

if __name__ == '__main__':
    model = SFDiscriminator()

    x = torch.randn(16, 1, 40, 80)
    print(x.shape)

    out = model(x)
    print(len(out), "Shape of output")

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
