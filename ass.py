from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir,name):
    logger = logging.getLogger(name)
    fhandler = logging.FileHandler(filename=file_dir)
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.view(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_CC(nn.Module):
    def __init__(self):
        super(Loss_CC, self).__init__()

    def forward(self, img1, img2):
        img1 = img1.contiguous().view(img1.size(0), img1.size(1), -1)
        img2 = img2.contiguous().view(img2.size(0), img1.size(1), -1)

        C1 = img1.size(2) * torch.sum(img1 * img2, 2) - torch.sum(img1, 2) * torch.sum(img2, 2)
        C2 = img1.size(2) * torch.sum(img1 ** 2, 2) - torch.sum(img1, 2) ** 2
        C3 = img2.size(2) * torch.sum(img2 ** 2, 2) - torch.sum(img2, 2) ** 2

        # 计算相关系数
        cc = C1 / ((C2 * C3) ** 0.5)

        return torch.mean(cc)


class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=1):
        error = im_fake - im_true
        sqrt_error = torch.pow(error, 2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        psnr=20*torch.log(data_range / rmse) / np.log(10.)
        return torch.mean(psnr)

class Loss_SAM(nn.Module):
    def __init__(self):
        super(Loss_SAM, self).__init__()

    def forward(self, im_fake, im_true):
        sum1 = torch.sum(im_true * im_fake, 1)
        sum2 = torch.sum(im_true * im_true, 1)
        sum3 = torch.sum(im_fake * im_fake, 1)
        t = (sum2 * sum3) ** 0.5
        numlocal = torch.gt(t, 0)
        num = torch.sum(numlocal)
        t = sum1 / t
        angle = torch.acos(t)
        sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
        if num == 0:
            averangle = sumangle
        else:
            averangle = sumangle / num
        SAM = averangle * 180 / 3.14159256
        return SAM

class Loss_ERGAS(nn.Module):
    def __init__(self):
        super(Loss_ERGAS, self).__init__()

    def forward(self, im_fake, im_true,ratio):
        summ = 0
        for i in range(im_true.shape[1]):
            a1 = torch.mean((im_true[:,i, :, :] - im_fake[:,i, :, :]) ** 2)
            m1 = torch.mean(im_true[:,i, :, :])
            a2 = m1 * m1
            summ = summ + a1 / a2
        ERGAS = 100 * (1 / ratio) * ((summ / im_true.shape[1]) ** 0.5)
        return ERGAS


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close