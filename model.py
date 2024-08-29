import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, normalize=False, downsample=False):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.normalize = normalize
        self.downsample = downsample
        self.learned_shortcut = dim_in != dim_out

        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)

        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)

        if self.learned_shortcut:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.conv1(self.actv(x))

        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.conv2(self.actv(x))
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s).view(s.size(0), -1, 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, upsample=False):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.upsample = upsample
        self.learned_shortcut = dim_in != dim_out

        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)

        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)

        if self.learned_shortcut:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_shortcut:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.actv(self.norm1(x, s))

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.conv2(self.actv(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s) + self._shortcut(x)
        return out / math.sqrt(2)


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size  # dim_in = 64

        self.from_rgb = nn.Conv2d(3, dim_in, 3, 1, 1)
        self.encode = []
        self.decode = []
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True), nn.LeakyReLU(0.2), nn.Conv2d(dim_in, 3, 1, 1, 0)
        )

        repeat_num = 4  # int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResidualBlock(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.append(AdainResidualBlock(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(ResidualBlock(dim_out, dim_out, normalize=True))
            self.decode.append(AdainResidualBlock(dim_out, dim_out, style_dim))

        self.encode = nn.ModuleList(self.encode)
        self.decode = nn.ModuleList(self.decode[::-1])

    def forward(self, x, s):
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        return self.to_rgb(x)


class MappingNetwork(nn.Module):  # F(noise, y) -> style in domain y
    def __init__(self, latent_dim=16, style_dim=64, num_domains=10):
        super().__init__()
        shared_layers = [nn.Linear(latent_dim, 512), nn.ReLU()]
        for _ in range(3):
            shared_layers += [nn.Linear(512, 512), nn.ReLU()]
        self.shared = nn.Sequential(*shared_layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            unshared_layers = []
            for _ in range(3):
                unshared_layers += [nn.Linear(512, 512), nn.ReLU()]
            unshared_layers += [nn.Linear(512, style_dim)]

            self.unshared.append(nn.Sequential(*unshared_layers))

    def forward(self, z, y):  # y = domain [batch]
        h = self.shared(z)
        # [batch, num_domains, style_dim]
        out = torch.stack([layer(h) for layer in self.unshared], dim=1)
        s = out[torch.arange(len(y), device=y.device), y]  # [batch, style_dim]
        return s


class StyleEncoder(nn.Module):  # E(img, y) -> style in domain y
    def __init__(self, img_size=256, style_dim=64, num_domains=10, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = 6  # int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResidualBlock(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2), nn.Conv2d(dim_out, dim_out, 4, 1, 0), nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)  # (batch, dim_out, 1x1)

        self.unshared = nn.ModuleList([nn.Linear(dim_out, style_dim) for _ in range(num_domains)])

    def forward(self, x, y):  # y = target domain [batch, 2]
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        # (batch, num_domains, style_dim)
        out = torch.stack([layer(h) for layer in self.unshared], dim=1)
        s = out[torch.arange(len(y), device=y.device), y]  # [batch, style_dim]
        return s


class Discriminator(nn.Module):  # D(img) -> real/fake? in each domain
    def __init__(self, img_size=256, num_domains=10, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = 6  # int(np.log2(img_size)) - 2
        for _ in range(repeat_num):  # 256x256 -> 4x4
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResidualBlock(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        blocks += [
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, dim_out, 4, 1, 0),  # 4x4 -> 1x1
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_out, num_domains, 1, 1, 0),
        ]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):  # y = target domain [batch]
        out = self.main(x)
        out = out.view(out.size(0), -1)  # [batch, num_domains]
        out = out[torch.arange(len(y), device=y.device), y]  # [batch]
        return out
