import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import os
import re
from model.HED import Hed


def crop_resize(image, bbox, imsize=64, cropsize=28, label=None):
    """"
    :param image: (b, 3, h, w)
    :param bbox: (b, o, 4)
    :param imsize: input image size
    :param cropsize: image size after crop
    :param label:
    :return: crop_images: (b*o, 3, h, w)
    """
    crop_images = list()
    b, o, _ = bbox.size()
    if label is not None:
        rlabel = list()
    for idx in range(b):
        for odx in range(o):
            if torch.min(bbox[idx, odx]) < 0:
                continue
            crop_image = image[idx:idx+1, :, int(imsize*bbox[idx, odx, 1]):int(imsize*(bbox[idx, odx, 1]+bbox[idx, odx, 3])),
                               int(imsize*bbox[idx, odx, 0]):int(imsize*(bbox[idx, odx, 0]+bbox[idx, odx, 2]))]
            crop_image = F.interpolate(crop_image, size=(cropsize, cropsize), mode='bilinear')
            crop_images.append(crop_image)
            if label is not None:
                rlabel.append(label[idx, odx, :].unsqueeze(0))
    if label is not None:
        return torch.cat(crop_images, dim=0), torch.cat(rlabel, dim=0)
    return torch.cat(crop_images, dim=0)


def truncted_random(num_o=8, thres=1.0):
    z = np.ones((1, num_o, 128)) * 100
    for i in range(num_o):
        for j in range(128):
            while z[0, i, j] > thres or z[0, i, j] < - thres:
                z[0, i, j] = np.random.normal()
    return z


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        ##
        if X.size(2) == 8:
            h_relu5 = torch.zeros_like(h_relu4)
        else:
            h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def load_model(model, model_dir=None, appendix=None, epoch='l'):

    load_epoch = None
    load_model = None

    if epoch == 's' or not os.path.isdir(model_dir) or len(os.listdir(model_dir)) == 0:
        load_epoch = 0
        if not os.path.isdir(model_dir):
            print('models dir not exist')
        elif len(os.listdir(model_dir)) == 0:
            print('models dir is empty')

        print('train from scratch.')
        return load_epoch, model

    # load latest epoch
    if epoch == 'l':
        for file in os.listdir(model_dir):
            if appendix is not None and appendix not in file:
                continue

            if file.endswith('.pth'):
                current_epoch= re.search('G_\d+', file).group(0).split('_')[1]

                if len(current_epoch) > 0:
                    current_epoch = int(current_epoch)

                    if load_epoch is None or current_epoch > load_epoch:
                        load_epoch = current_epoch
                        load_model = os.path.join(model_dir, file)
                else:
                    continue

        print('load from epoch: %d' % load_epoch)
        model.load_state_dict(torch.load(load_model))

        return load_epoch, model
    # from given epoch
    else:
        epoch = int(epoch)
        for file in os.listdir(model_dir):
            if file.endswith('.pth'):
                current_epoch= re.search('G_\d+', file).group(0).split('_')[1]
                if len(current_epoch) > 0:
                    if int(current_epoch) == epoch:
                        load_epoch = epoch
                        load_model = os.path.join(model_dir, file)
                        break
        if load_model:
            model.load_state_dict(torch.load(load_model))
            print('load from epoch: %d' % load_epoch)
        else:
            load_epoch = 0
            print('there is not saved models of epoch %d' % epoch)
            print('train from scratch.')
        return load_epoch, model


def loadHed():
    hed_model = Hed()
    save_path = './model/35.pth'
    hed_model.load_state_dict(torch.load(save_path))
    for param in hed_model.parameters():
        param.requires_grad = False
    return hed_model