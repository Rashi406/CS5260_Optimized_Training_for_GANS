"""
Code for displaying intermediate training results.
The code was adapted from the github repository by Hyeonwoo Kang
on 27/04/2022.
The github repository for original code is available on:
https://github.com/znxlwm/pytorch-pix2pix
The code is based on the paper:
Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." 
arXiv preprint arXiv:1611.07004 (2016).
Paper:https://arxiv.org/pdf/1611.07004.pdf
"""

import itertools, torch
import matplotlib.pyplot as plt

def show_result(G, x_, y_, num_epoch, show = True, save = False, path = 'result.png'):
    G.eval()
    test_images = G(x_).type(torch.float64)
    
    size_figure_grid = 3
    fig, ax = plt.subplots(x_.size()[0], size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(x_.size()[0]), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    
    for i in range(x_.size()[0]):
        ax[i, 0].cla()
        ax[i, 0].imshow((x_[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 1].cla()
        ax[i, 1].imshow((test_images[i].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
        ax[i, 2].cla()
        ax[i, 2].imshow((y_[i].numpy().transpose(1, 2, 0) + 1) / 2)
    
    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    
    if save:
        plt.savefig(path)

    if show:
        
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def display_progress(cond, fake, real, figsize=(10,5)):
    cond = cond.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    real = real.detach().cpu().permute(1, 2, 0)
    
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(cond)
    ax[2].imshow(fake)
    ax[1].imshow(real)
    plt.show()
