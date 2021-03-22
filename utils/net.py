import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.spectral_norm import spectral_norm as SpectralNorm


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input


class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        out = self.main(x)
        return out


class Generator_makeup(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    # input 2 images and output 2 images as well
    def __init__(self, conv_dim=64, repeat_num=6, input_nc=6):
        super(Generator_makeup, self).__init__()

        layers = []
        layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers_1 = []
        layers_1.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.Tanh())
        self.branch_1 = nn.Sequential(*layers_1)
        layers_2 = []
        layers_2.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_2.append(nn.Tanh())
        self.branch_2 = nn.Sequential(*layers_2)

    def forward(self, x, y):
        input_x = torch.cat((x, y), dim=1)
        out = self.main(input_x)
        out_A = self.branch_1(out)
        out_B = self.branch_2(out)
        return out_A, out_B


class Generator_branch(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    # input 2 images and output 2 images as well
    def __init__(self, conv_dim=64, repeat_num=6, input_nc=3):
        super(Generator_branch, self).__init__()

        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        layers_branch.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_0 = nn.Sequential(*layers_branch)

        # Branch input
        layers_branch = []
        layers_branch.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        layers_branch.append(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers_branch.append(nn.InstanceNorm2d(conv_dim * 2, affine=True))
        layers_branch.append(nn.ReLU(inplace=True))
        self.Branch_1 = nn.Sequential(*layers_branch)

        # Down-Sampling, branch merge
        layers = []
        curr_dim = conv_dim * 2
        layers.append(
            nn.Conv2d(
                curr_dim * 2,
                curr_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )
        layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(
                nn.ConvTranspose2d(
                    curr_dim,
                    curr_dim // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers_1 = []
        layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_1.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_1.append(nn.ReLU(inplace=True))
        layers_1.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_1.append(nn.Tanh())
        self.branch_1 = nn.Sequential(*layers_1)
        layers_2 = []
        layers_2.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_2.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        layers_2.append(nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers_2.append(nn.InstanceNorm2d(curr_dim, affine=True))
        layers_2.append(nn.ReLU(inplace=True))
        layers_2.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers_2.append(nn.Tanh())
        self.branch_2 = nn.Sequential(*layers_2)

    def forward(self, x, y):
        input_x = self.Branch_0(x)
        input_y = self.Branch_1(y)
        input_fuse = torch.cat((input_x, input_y), dim=1)
        out = self.main(input_fuse)
        out_A = self.branch_1(out)
        out_B = self.branch_2(out)
        return out_A, out_B


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, repeat_num=3, norm="SN"):
        super(Discriminator, self).__init__()

        layers = []
        if norm == "SN":
            layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        else:
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if norm == "SN":
                layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        # k_size = int(image_size / np.power(2, repeat_num))
        if norm == "SN":
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1)))
        else:
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        if norm == "SN":
            self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, padding=1, bias=False)

        # conv1 remain the last square size, 256*256-->30*30
        # self.conv2 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=False))
        # conv2 output a single number

    def forward(self, x):
        h = self.main(x)
        # out_real = self.conv1(h)
        out_makeup = self.conv1(h)
        # return out_real.squeeze(), out_makeup.squeeze()
        return out_makeup.squeeze()


class VGG(nn.Module):
    def __init__(self, pool="max"):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == "max":
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == "avg":
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out["r11"] = F.relu(self.conv1_1(x))
        out["r12"] = F.relu(self.conv1_2(out["r11"]))
        out["p1"] = self.pool1(out["r12"])
        out["r21"] = F.relu(self.conv2_1(out["p1"]))
        out["r22"] = F.relu(self.conv2_2(out["r21"]))
        out["p2"] = self.pool2(out["r22"])
        out["r31"] = F.relu(self.conv3_1(out["p2"]))
        out["r32"] = F.relu(self.conv3_2(out["r31"]))
        out["r33"] = F.relu(self.conv3_3(out["r32"]))
        out["r34"] = F.relu(self.conv3_4(out["r33"]))
        out["p3"] = self.pool3(out["r34"])
        out["r41"] = F.relu(self.conv4_1(out["p3"]))

        out["r42"] = F.relu(self.conv4_2(out["r41"]))
        out["r43"] = F.relu(self.conv4_3(out["r42"]))
        out["r44"] = F.relu(self.conv4_4(out["r43"]))
        out["p4"] = self.pool4(out["r44"])
        out["r51"] = F.relu(self.conv5_1(out["p4"]))
        out["r52"] = F.relu(self.conv5_2(out["r51"]))
        out["r53"] = F.relu(self.conv5_3(out["r52"]))
        out["r54"] = F.relu(self.conv5_4(out["r53"]))
        out["p5"] = self.pool5(out["r54"])

        return [out[key] for key in out_keys]
