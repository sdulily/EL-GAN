import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_v2 import *
from utils.util import *
import imageio
import torchvision.utils as vutils
from skimage import draw
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_dataloader(dataset = 'coco', img_size=128):
    if dataset == 'coco':
        dataset = CocoSceneGraphDataset(image_dir='./datasets/coco/images/val2017/',
                                        instances_json='./datasets/coco/annotations/instances_val2017.json',
                                        stuff_json='./datasets/coco/annotations/stuff_val2017.json',
                                        stuff_only=True, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'vg':
        with open("./datasets/vg/vocab.json", "r") as read_file:
            vocab = json.load(read_file)
        dataset = VgSceneGraphDataset(vocab=vocab,
                                      h5_path='./datasets/vg/val.h5',
                                      image_dir='./datasets/vg/images/',
                                      image_size=(img_size, img_size), left_right_flip=False, max_objects=30)

    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1,
                    drop_last=True, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 31

    dataloader = get_dataloader(args.dataset)

    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()
    hed = loadHed().cuda()

    if not os.path.isfile(args.model_path):
        return
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    thres=2.0

    fake_dir = '{save_path}/fake'.format(save_path=args.sample_path)
    real_dir = '{save_path}/real'.format(save_path=args.sample_path)
    if not Path(fake_dir).exists(): Path(fake_dir).mkdir(parents=True)
    if not Path(real_dir).exists(): Path(real_dir).mkdir(parents=True)

    for idx, data in enumerate(dataloader):
        real_images, label, bbox, _, _, _, _ = data

        real_images, label = real_images.cuda(), label.long().unsqueeze(-1).cuda()
        z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()

        fake_images, _, _, _, _, _ = netG.forward(hed, z_obj, bbox.cuda(), z_im, label.squeeze(dim=-1))

        imageio.imwrite("{fake_dir}/sample_{idx}.jpg".format(fake_dir=fake_dir, idx=idx),
                        fake_images[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
        imageio.imwrite("{real_dir}/real_{idx}.jpg".format(real_dir=real_dir, idx=idx),
                        real_images[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)

    print('over!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vg',
                        help='training dataset')
    parser.add_argument('--model_path', type=str, default='./pretrained/coco128.pth',
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='./samples/coco128',
                        help='path to save generated images')
    args = parser.parse_args()
    main(args)
