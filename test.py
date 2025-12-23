import argparse, os
import scipy.io as sio
import torch
import math
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset_mh import DatasetFromHdf5
from ass import Loss_MRAE,SSIM,Loss_SAM,initialize_logger,AverageMeter,Loss_PSNR,Loss_RMSE,Loss_ERGAS,Loss_CC
from pngsave import torch2png
from store2tiff import writeTiff as wtiff
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append("..")

# Testing
parser = argparse.ArgumentParser(description="PyTorch Test")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda?")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--gpus", default="8", type=str, help="gpu ids (default: 0)")

#main fuction
opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

test_set = DatasetFromHdf5("data/Sen2Chikusei_test.h5",False)
name='Sen2Chikusei/PoXnet_b1_adamax_L1loss_lr0.0001'
checkpoint = torch.load('/home/jianghe/13MambaFus/checkpoint/'+name+'/model_epoch_200.pth', map_location=torch.device('cpu'))

test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print("===> Building model")
ratio=4

criterion_cc = Loss_CC()
criterion_psnr = Loss_PSNR()
criterion_ssim = SSIM()
criterion_sam= Loss_SAM()
criterion_ergas= Loss_ERGAS()

losses_cc = AverageMeter()
losses_psnr = AverageMeter()
losses_ssim = AverageMeter()
losses_sam = AverageMeter()
losses_ergas = AverageMeter()

model=checkpoint['model']

def torch2tiff(torch,filename):
    out_temp = torch.cpu()
    output_temp = out_temp.numpy().astype(np.float32)
    output_temp = np.transpose(output_temp, [1, 2, 0])
    output = output_temp
    wtiff(output, output.shape[2], output.shape[0], output.shape[1], filename)

def validate(val_loader, model):
    with torch.no_grad():

        for i,batch in enumerate(val_loader,0):
            mul, pan, target_data = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

            if opt.cuda:
                mul = mul.cuda()
                pan = pan.cuda()
                target_data = target_data.cuda()
            # # compute output
            output = model(pan,mul)

            cc = criterion_cc(output, target_data)
            psnr = criterion_psnr(output * 255, target_data * 255, 255)
            ssims = criterion_ssim(output, target_data)
            sam = criterion_sam(output, target_data)
            ergars = criterion_ergas(output, target_data, 4)

            losses_cc.update(cc.data)
            losses_psnr.update(psnr.data)
            losses_ssim.update(ssims.data)
            losses_sam.update(sam.data)
            losses_ergas.update(ergars.data)

            imagepath='output/'+name+'/'
            if not os.path.exists(imagepath):
                os.makedirs(imagepath)

            # output_fus=reconstruct_tensor(output,target.shape,1200)
            print(output.shape)
            torch2tiff(mul_data[0, :, :, :], imagepath+'Tiff'+str(i)+'mul.tiff')
            torch2tiff(pan_data[0, :, :, :], imagepath+'Tiff'+str(i)+'pan.tiff')
            torch2tiff(output[0, :, :, :], imagepath+'Tiff'+str(i)+'output.tiff')
            torch2tiff(target_data[0, :, :, :], imagepath + 'Tiff' + str(i) + 'gt.tiff')
        print("     Test done! CC {:.6f}, PSNR {:.6f}, SSIM {:.6f}, SAM {:.6f}, ERGAS {:.6f}".format(losses_cc.avg,
                                                                                                   losses_psnr.avg,
                                                                                                   losses_ssim.avg,
                                                                                                   losses_sam.avg,
                                                                                                   losses_ergas.avg,
                                                                                                   ))

print("===> Setting GPU")
model.cuda()
model.eval()

with torch.no_grad():
    validate(test_data_loader, model)











