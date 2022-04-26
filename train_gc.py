"""
Code for training Pix2Pix model using gradient clipping.
The code was adapted from the github repository by Hyeonwoo Kang
on 27/04/2022.
The github repository for original code is available on:
https://github.com/znxlwm/pytorch-pix2pix
The code is based on the paper:
Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." 
arXiv preprint arXiv:1611.07004 (2016).
Paper:https://arxiv.org/pdf/1611.07004.pdf
"""

import os
from pathlib import Path
import math
import colossalai
import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn.metric import Accuracy
from colossalai.utils import MultiTimer, get_dataloader
from tqdm import tqdm
import time, pickle
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Dataload import *
from model import *
from Display import *
from copy import deepcopy

def main():
    colossalai.launch_from_torch(config='./config_gc.py')
    logger = get_dist_logger()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_dataset = data_load("facades/", "train", transform)

    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=gpc.config.BATCH_SIZE,
                                      pin_memory=True,
                                      )

    test_dataset = data_load('facades/', "val", transform)

    test_dataloader = get_dataloader(dataset=test_dataset,
                                      add_sampler=False,
                                      batch_size=gpc.config.tst_batch_size,
                                      pin_memory=True,
                                      )

    test = test_dataloader.__iter__().__next__()[0]
    img_size = test.size()[2]
    if gpc.config.inverse_order:
        fixed_y_ = test[:, :, :, 0:img_size]
        fixed_x_ = test[:, :, :, img_size:]
    else:
        fixed_x_ = test[:, :, :, 0:img_size]
        fixed_y_ = test[:, :, :, img_size:]

    if img_size != gpc.config.input_size:
        fixed_x_ = imgs_resize(fixed_x_, gpc.config.input_size)
        fixed_y_ = imgs_resize(fixed_y_, gpc.config.input_size)

    g_model = generator(gpc.config.ngf)
    d_model = discriminator(gpc.config.ndf)
    g_model.weight_init(mean=0.0, std=0.02)
    d_model.weight_init(mean=0.0, std=0.02)
    g_model.cuda()
    d_model.cuda()

    BCE_logit_loss = nn.BCEWithLogitsLoss().cuda()
    L1_loss = nn.L1Loss().cuda()    

    G_optimizer = optim.Adam(g_model.parameters(), lr=gpc.config.lrG, betas=(gpc.config.beta1, gpc.config.beta2))
    D_optimizer = optim.Adam(d_model.parameters(), lr=gpc.config.lrD, betas=(gpc.config.beta1, gpc.config.beta2))

    G,_,_, _ = colossalai.initialize(g_model, G_optimizer, L1_loss)

    D, train_dataloader, test_dataloader, _ = colossalai.initialize(d_model,
                                  D_optimizer,
                                  BCE_logit_loss,
                                  train_dataloader,
                                  test_dataloader)
    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('training start!')
    start_time = time.time()
    for epoch in range(gpc.config.NUM_EPOCHS):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        num_iter = 0
        G.train()
        D.train()
        for x_, _ in train_dataloader:
            # train discriminator D        
            D.zero_grad()

            if gpc.config.inverse_order:
                y_ = x_[:, :, :, 0:img_size]
                x_ = x_[:, :, :, img_size:]
            else:
                y_ = x_[:, :, :, img_size:]
                x_ = x_[:, :, :, 0:img_size]
                
            if img_size != gpc.config.input_size:
                x_ = imgs_resize(x_, gpc.config.input_size)
                y_ = imgs_resize(y_, gpc.config.input_size)

            if gpc.config.resize_scale:
                x_ = imgs_resize(x_, gpc.config.resize_scale)
                y_ = imgs_resize(y_, gpc.config.resize_scale)

            if gpc.config.crop_size:
                x_, y_ = random_crop(x_, y_, gpc.config.crop_size)

            if gpc.config.fliplr:
                x_, y_ = random_fliplr(x_, y_)

            x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())
            
            G_result = G(x_).detach()

            ip_x = torch.vstack((x_,x_))
            ip_y = torch.vstack((y_,G_result))
            D_result = D(ip_x, ip_y).squeeze()
            op = torch.vstack((Variable(torch.ones(x_.shape[0],D_result.shape[1],D_result.shape[2]).cuda()), Variable(torch.zeros(x_.shape[0],D_result.shape[1],D_result.shape[2]).cuda())))
            
            D_train_loss = D.criterion(D_result,op)            
            D.backward(D_train_loss)
            D.step()

            train_hist['D_losses'].append(D_train_loss.data.item())

            D_losses.append(D_train_loss.data.item())
            
            # train generator G
            G.zero_grad()

            G_result = G(x_)
            D_result = D(x_, G_result).squeeze()

            G_train_loss = G.criterion(D_result, Variable(torch.ones(D_result.size()).cuda())) + gpc.config.L1_lambda * L1_loss(G_result, y_)
            G.backward(G_train_loss)
            G.step()

            train_hist['G_losses'].append(G_train_loss.data.item())

            G_losses.append(G_train_loss.data.item())

            num_iter += 1

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), gpc.config.NUM_EPOCHS, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),  torch.mean(torch.FloatTensor(G_losses))))
        if epoch%5 == 0:
            show_result(G, Variable(fixed_x_.cuda(), requires_grad=False), fixed_y_, (epoch+1), save=True)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), gpc.config.NUM_EPOCHS, total_ptime))
    print("Training finish!... save training results")

    show_train_hist(train_hist, save=False, path='train_hist.png')

if __name__ == '__main__':
    main()

