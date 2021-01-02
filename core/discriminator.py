import torch.nn as nn
from utils.util import weights_init


class Dblock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 256):
        super(Dblock, self).__init__()
        self.input_spec = nn.utils.weight_norm(nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=1))
        self.input_cond = nn.utils.weight_norm(nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1))

        self.output_spec = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.utils.weight_norm(nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=1)),
        )

        self.output_cond = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.utils.weight_norm(nn.Conv1d(output_dim, hidden_dim, kernel_size=3, stride=1)),
        )

    def forward(self, spec, cond):
        x1 = self.input_spec(spec)
        y1 = self.input_cond(cond)
        x = x1 + y1
        out1 = self.output_spec(x)
        out2 = self.output_cond(y1)
        return out1 + x1, out2 + cond

class Discriminator(nn.Module):
    def __init__(self, nf = 80, n_layers = 4, hidden_dim = 256):
        super(Discriminator, self).__init__()
        discriminator = nn.ModuleDict()

        for n in range(1, n_layers + 1):
            discriminator["layer_%d" % n] = Dblock(nf, nf*2, hidden_dim)
            nf = nf * 2

        self.disc_spec = nn.utils.weight_norm(nn.Conv1d(nf, 1, kernel_size=3, stride=1, padding=1))


        self.disc_cond = nn.utils.weight_norm(nn.Conv1d(
            hidden_dim, 1, kernel_size=3, stride=1, padding=1
        ))
        self.discriminator = discriminator

    def forward(self, x, y):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()

        for key, module in self.discriminator.items():
            x, y = module(x, y)
            features.append(x)
        out  = self.disc_spec(x) + self.disc_cond(y)
        features.append(out)
        return features[:-1], features[-1]


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_D = 3, ndf = 80, n_layers = 4, downsampling_factor = 2, hidden_dim = 256):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = Discriminator(
                ndf, n_layers, hidden_dim
            )

        self.downsample = nn.AvgPool1d(downsampling_factor, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x, y):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x, y))
            x = self.downsample(x)
            y = self.downsample(y)
        return results

