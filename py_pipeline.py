"""
Code for training Pix2Pix model using model parallelism.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time, pickle
from colossalai.core import global_context as gpc
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Dataload import *
from model import *
from Display import *
from config import *

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_dataset = data_load("facades/", "train", transform)

    train_dataloader = DataLoader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=BATCH_SIZE,
                                      )

    test_dataset = data_load('facades/', "val", transform)

    test_dataloader = DataLoader(dataset=test_dataset,
                                      batch_size=tst_batch_size,
                                      )

    test = test_dataloader.__iter__().__next__()[0]
    img_size = test.size()[2]
    if inverse_order:
        fixed_y_ = test[:, :, :, 0:img_size]
        fixed_x_ = test[:, :, :, img_size:]
    else:
        fixed_x_ = test[:, :, :, 0:img_size]
        fixed_y_ = test[:, :, :, img_size:]

    if img_size != input_size:
        fixed_x_ = imgs_resize(fixed_x_, input_size)
        fixed_y_ = imgs_resize(fixed_y_, input_size)

    G = generator(ngf)
    D = discriminator(ndf)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda('cuda:0')
    D.cuda('cuda:1')

    BCE_loss = nn.BCEWithLogitsLoss()
    #BCE_loss0 = nn.BCEWithLogitsLoss().to('cuda:0')

    L1_loss = nn.L1Loss()

    G_optimizer = optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=lrD, betas=(beta1, beta2))
    

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('training start!')
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        num_iter = 0
        G.train(True)
        D.train(True)
        for x_, _ in train_dataloader:
            # train discriminator D        
            D_optimizer.zero_grad()

            if inverse_order:
                y_ = x_[:, :, :, 0:img_size]
                x_ = x_[:, :, :, img_size:]
            else:
                y_ = x_[:, :, :, img_size:]
                x_ = x_[:, :, :, 0:img_size]
                
            if img_size != input_size:
                x_ = imgs_resize(x_, input_size)
                y_ = imgs_resize(y_, input_size)

            if resize_scale:
                x_ = imgs_resize(x_, resize_scale)
                y_ = imgs_resize(y_, resize_scale)

            if crop_size:
                x_, y_ = random_crop(x_, y_, crop_size)

            if fliplr:
                x_, y_ = random_fliplr(x_, y_)


            #x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())
            
            D_result_real = D(Variable(x_.cuda('cuda:1')), Variable(y_.cuda('cuda:1'))).squeeze()
            D_real_loss = BCE_loss(D_result_real, Variable(torch.ones(D_result_real.size()).cuda('cuda:1')))

            G_result = G(Variable(x_.cuda('cuda:0'))).detach().cuda('cuda:1')
            D_result_fake = D(Variable(x_.cuda('cuda:1')), G_result).squeeze()
            D_fake_loss = BCE_loss(D_result_fake, Variable(torch.zeros(D_result_fake.size()).cuda('cuda:1')))

            #ip_x = torch.vstack((Variable(x_.to('cuda:1')),Variable(x_.to('cuda:1'))))
            #ip_y = torch.vstack((y_,G_result))
            #D_result = D(ip_x, ip_y).squeeze()
            #op = torch.vstack((Variable(torch.ones(x_.shape[0],D_result.shape[1],D_result.shape[2]).cuda()), Variable(torch.zeros(x_.shape[0],D_result.shape[1],D_result.shape[2]).cuda())))
            #D_train_loss = D.criterion(D_result,op)

            D_train_loss = (D_real_loss + D_fake_loss) * 0.5
            D_train_loss.backward()
            D_optimizer.step()

            train_hist['D_losses'].append(D_train_loss.detach().item())

            D_losses.append(D_train_loss.detach().item())
            # train generator G
            
            G_optimizer.zero_grad()

            G_result = G(Variable(x_.cuda('cuda:0'))).cuda('cuda:1')
            D_result = D(Variable(x_.cuda('cuda:1')), G_result).squeeze().cuda('cuda:0')

            G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).cuda('cuda:0'))) + L1_lambda * L1_loss(G_result.cuda('cuda:0'), Variable(y_.cuda('cuda:0')))
            G_train_loss.backward()
            G_optimizer.step()

            train_hist['G_losses'].append(G_train_loss.detach().item())

            G_losses.append(G_train_loss.detach().item())
            
            num_iter += 1

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), NUM_EPOCHS, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),  torch.mean(torch.FloatTensor(G_losses))))
        if epoch%5 == 0:
            show_result(G, Variable(fixed_x_.cuda('cuda:0'), requires_grad=False), fixed_y_, (epoch+1), save=True)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), NUM_EPOCHS, total_ptime))
    print("Training finish!... save training results")

    show_train_hist(train_hist, save=True, path='train_hist.png')

if __name__ == '__main__':
    main()
