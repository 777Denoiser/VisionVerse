# utility function for handling CLIP, SIREN, and image manipulation.

import torch
from torch import nn
import math


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-math.sqrt(6 / self.in_features) / self.omega_0,
                                            math.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SIREN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0_initial, w0=30.):
        super().__init__()
        layers = []
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(SineLayer(
                layer_dim_in,
                dim_hidden,
                is_first=is_first,
                omega_0=layer_w0
            ))

        self.net = nn.Sequential(*layers)
        self.last_layer = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)


def get_init_images(size):
    coords = torch.ones(1, 3, size[0], size[1])
    coords[:, 0, :, :] = torch.linspace(-1, 1, size[0])[None, :]
    coords[:, 1, :, :] = torch.linspace(-1, 1, size[1])[:, None]
    return coords

