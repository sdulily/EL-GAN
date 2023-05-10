#coding=gbk
import argparse
import os
import pickle
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from utils.util import *
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_v2 import *
from model.rcnn_discriminator import *
from model.sync_batchnorm import DataParallelWithCallback
from utils.logger import setup_logger
from scipy import misc
from focal_frequency_loss import FocalFrequencyLoss as FFL
import torch.multiprocessing as mp
import torch.distributed as dist
import warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def get_dataset(dataset, img_size):
    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir='/mnt/datasets/coco/images/train2017/',
                                        instances_json='/mnt/datasets/coco/annotations/instances_train2017.json',
                                        stuff_json='/mnt/datasets/coco/annotations/stuff_train2017.json',
                                        stuff_only=True, image_size=(img_size, img_size), left_right_flip=True)
    elif dataset == 'vg':
        vocab_json = os.path.join('/mnt/datasets/vg/vocab.json')
        with open(vocab_json, 'r') as f:
            vocab = json.load(f)
        data = VgSceneGraphDataset(vocab=vocab, h5_path='/mnt/datasets/vg/train.h5',
                                      image_dir='/mnt/datasets/vg/images/',
                                      image_size=(img_size, img_size), max_objects=30, left_right_flip=True)
    return data


def main(gpu, args):
    # parameters
    img_size = 128
    z_dim = 128
    lamb_obj = 1.0
    lamb_img = 0.1
    num_classes = 184 if args.dataset == 'coco' else 179
    num_obj = 8 if args.dataset == 'coco' else 31

    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    # data loader
    train_data = get_dataset(args.dataset, img_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, num_replicas=args.world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        drop_last=True, shuffle=False, num_workers=4, sampler=train_sampler)

    # Load model
    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3)
    netD = CombineDiscriminator128_addFreq(num_classes=num_classes)
    hed = loadHed()

    torch.cuda.set_device(gpu)
    netG.cuda()
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu], find_unused_parameters=True)
    netD.cuda()
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu], find_unused_parameters=True)
    hed.cuda()

    g_lr, d_lr = args.g_lr, args.d_lr
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key:
                gen_parameters += [{'params': [value], 'lr': g_lr * 0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]

    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

    # make dirs
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.mkdir(os.path.join(args.out_path, 'model/'))
    if rank == 0:
        writer_all = SummaryWriter(os.path.join(args.out_path, 'log'))
        logger = setup_logger("EL-GAN", args.out_path, 0)
        logger.info(netG)
        logger.info(netD)

    start_time = time.time()
    vgg_loss = VGGLoss()
    ffl = FFL(loss_weight=1.0, alpha=1.0)

    start_epoch, netG = load_model(netG, model_dir=os.path.join(args.out_path, 'model'), appendix='G', epoch=args.resume_epoch)

    if start_epoch < args.total_epoch:
        for epoch in range(start_epoch, args.total_epoch):
            print('epoch: ', epoch)
            netG.train()
            netD.train()

            for idx, data in enumerate(dataloader):

                train_sampler.set_epoch(epoch)
                real_images, label, bbox, real_images8, real_images16, real_images32, real_images64 = data
                real_images, label, bbox = real_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float()
                real_images8, real_images16, real_images32, real_images64 = real_images8.cuda(), real_images16.cuda(), real_images32.cuda(), real_images64.cuda()

                # update D network
                netD.zero_grad()
                real_images, label = real_images.cuda(), label.long().cuda()
                d_out_real, d_out_robj, d_out_rfreq = netD(real_images, bbox, label)
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()
                d_loss_rfreq = torch.nn.ReLU()(1.0 - d_out_rfreq).mean()

                z = torch.randn(real_images.size(0), num_obj, z_dim).cuda()
                fake_images, mid_img1, mid_img2, mid_img3, mid_img4, fake_edge = netG(hed, z, bbox, y=label.squeeze(dim=-1))
                d_out_fake, d_out_fobj, d_out_ffreq = netD(fake_images.detach(), bbox, label)
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()
                d_loss_ffreq = torch.nn.ReLU()(1.0 + d_out_ffreq).mean()

                d_loss = lamb_obj * (d_loss_robj + d_loss_fobj + d_loss_rfreq) + lamb_img * (d_loss_real + d_loss_fake + d_loss_ffreq)
                d_loss.backward()
                d_optimizer.step()

                D_loss = {}
                D_loss['D/loss'] = d_loss.item()
                D_loss['D/d_loss_robj'] = d_loss_robj.item()
                D_loss['D/d_loss_fobj'] = d_loss_fobj.item()
                D_loss['D/d_loss_real'] = d_loss_real.item()
                D_loss['D/d_loss_fake'] = d_loss_fake.item()
                D_loss['D/d_loss_rfreq'] = d_loss_rfreq.item()
                D_loss['D/d_loss_ffreq'] = d_loss_ffreq.item()

                # update G network
                if (idx % 1) == 0:
                    netG.zero_grad()
                    g_out_fake, g_out_obj, g_out_ffreq = netD(fake_images, bbox, label)
                    g_loss_fake = - g_out_fake.mean()
                    g_loss_obj = - g_out_obj.mean()
                    g_loss_ffreq = - g_out_ffreq.mean()

                    feat_loss = vgg_loss(fake_images, real_images).mean()
                    ffl_loss = ffl(fake_images, real_images).mean()

                    feat_loss8 = vgg_loss(mid_img1, real_images8).mean()
                    feat_loss16 = vgg_loss(mid_img2, real_images16).mean()
                    feat_loss32 = vgg_loss(mid_img3, real_images32).mean()
                    feat_loss64 = vgg_loss(mid_img4, real_images64).mean()

                    g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img + feat_loss + \
                             feat_loss8 + feat_loss16 + feat_loss32 + feat_loss64 + ffl_loss + g_loss_ffreq
                    g_loss.backward()
                    g_optimizer.step()

                    G_loss = {}
                    G_loss['G/loss'] = g_loss.item()
                    G_loss['G/g_loss_obj'] = g_loss_obj.item()
                    G_loss['G/g_loss_fake'] = g_loss_fake.item()
                    G_loss['G/feat_loss'] = feat_loss.item()
                    G_loss['G/feat_loss8'] = feat_loss8.item()
                    G_loss['G/feat_loss16'] = feat_loss16.item()
                    G_loss['G/feat_loss32'] = feat_loss32.item()
                    G_loss['G/feat_loss64'] = feat_loss64.item()
                    G_loss['G/ffl_loss'] = ffl_loss.item()
                    G_loss['G/g_loss_ffreq'] = g_loss_ffreq.item()

                    loss = dict(D_loss, **G_loss)

                if (idx+1) % 500 == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    if rank == 0:
                        logger.info("Step[{}/{}], Time Elapsed: [{}]".format(epoch + 1, idx + 1, elapsed))
                        writer_all.add_scalars('loss', loss, epoch * len(dataloader) + idx + 1)

            # save model
            if (epoch + 1) % 5 == 0 and rank == 0:
                torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch+1)))

    writer_all.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vg',
                        help='training dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size of training data. Default: 32') # 128
    parser.add_argument('--total_epoch', type=int, default=200,
                        help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str, default='./outputs_VG128/',
                        help='path to output files')
    parser.add_argument('--sample_path', type=str, default='./outputs_VG128/samples',
                        help='path to save generated images')
    parser.add_argument('--resume_epoch', type=str, default='l',
                        help='l: from latest; s: from scratch; xxx: from epoch xxx')

    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=4, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(main, nprocs=args.gpus, args=(args,))
