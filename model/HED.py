import torch.nn as nn
import torchvision.models as models
import torch
import torchvision


class Hed(nn.Module):
    def __init__(self):
        super(Hed, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = self.vgg16.features
        for param in self.vgg16.parameters():
            param.requires_grad = False

        self.score_dsn1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn4 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)
        self.score_dsn5 = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0)

        self.fuse = nn.Conv2d(5, 1, kernel_size=1, stride=1, padding=0)

        self.upsample2 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(1, 1, kernel_size=6, stride=4, padding=1)
        self.upsample4 = nn.ConvTranspose2d(1, 1, kernel_size=12, stride=8, padding=2)
        self.upsample5 = nn.ConvTranspose2d(1, 1, kernel_size=24, stride=16, padding=4)

        self.actv = nn.Sigmoid()

    def forward(self, x):
        cnt = 1
        for l in self.vgg16:
            x = l(x)
            if cnt == 4:
                y = self.score_dsn1(x)
                y = self.actv(y)
                return y
            cnt += 1

