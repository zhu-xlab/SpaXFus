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
from ass import SSIM,Loss_SAM,initialize_logger,AverageMeter,Loss_PSNR,Loss_RMSE,Loss_ERGAS,Loss_CC
from pngsave import torch2png
from store2tiff import writeTiff as wtiff
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from torch.utils.tensorboard import SummaryWriter
sys.path.append("..")

# Training
parser = argparse.ArgumentParser(description="PyTorch DTer")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--accumulation-steps", type=int, default=1, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10,help="Change the learning rate, Default: n=10")
parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda?")
parser.add_argument("--resume", default=r"", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="8", type=str, help="gpu ids (default: 0)")

#main fuction
opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

print("===> Loading datasets")
train_set = DatasetFromHdf5("/home/jianghe/13MambaFus/data/Sen2Chikusei_train.h5",True)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

test_set = DatasetFromHdf5("/home/jianghe/13MambaFus/data/Sen2Chikusei_test.h5",False)
test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print("===> Building model")
ratio=4

mschannel=4
hschannel=128

from SpaXFus_model import SpaXFus

model =SpaXFus(4,mschannel,hschannel,48,1)

opt_chosed = 'adamax'
filepath = "checkpoint/Sen2Chikusei/Mamba4ever_multifus_smam4_mid48_b" + str(opt.batchSize * opt.accumulation_steps) + "_" + opt_chosed + "_L1loss_lr" + str(opt.lr) + "/"


total_iteration = len(training_data_loader)*opt.nEpochs/opt.accumulation_steps
print("===> Setting Optimizer")
opt_Adam = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay) # lr=0.001
opt_SGD = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
opt_RMSprop = optim.RMSprop(model.parameters(), lr=opt.lr, alpha=0.9, weight_decay=opt.weight_decay) # lr=0.01
opt_Adamax = optim.Adamax(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) # lr=0.002
opt_dic = {'adam': opt_Adam, 'sgd': opt_SGD, 'rmsp': opt_RMSprop, 'adamax': opt_Adamax}
optimizer = opt_dic[opt_chosed]

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr,total_steps=int(total_iteration),pct_start=0.1)

criterion = nn.L1Loss(reduction='mean')
# criterion = nn.MSELoss(reduction='mean')
print(filepath)
if not os.path.exists(filepath):
    os.makedirs(filepath)

outfile = 'output' + filepath[filepath.index('/'):]
print(outfile)
if not os.path.exists(outfile):
    os.makedirs(outfile)

log_dir_U = os.path.join(filepath, 'train.log')
logger_train = initialize_logger(log_dir_U,'log_train')
log_dir_D = os.path.join(filepath, 'test.log')
logger_test = initialize_logger(log_dir_D,'log_test')
writer = SummaryWriter(os.path.join(filepath, 'vis'))
s = sum([np.prod(list(p.size())) for p in model.parameters()])
print('Number of params: %d' % s)

cuda = opt.cuda
if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
cudnn.benchmark = True

criterion_cc = Loss_CC()
criterion_psnr = Loss_PSNR()
criterion_ssim = SSIM()
criterion_sam= Loss_SAM()
criterion_ergas= Loss_ERGAS()

def torch2tiff(torch,filename):
    out_temp = torch.cpu()
    output_temp = out_temp.numpy().astype(np.float32)
    output_temp = np.transpose(output_temp, [1, 2, 0])
    output = output_temp
    wtiff(output, output.shape[2], output.shape[0], output.shape[1], filename)

def train(training_data_loader,optimizer, model, criterion, epoch, accumulation_steps,scheduler,loss_vaule):
    model.train()
    for iteration, batch in enumerate(training_data_loader, 0):
        mul,pan,target = Variable(batch[0]),Variable(batch[1]),Variable(batch[2])

        starttime = time.time()
        if opt.cuda:
            mul = mul.cuda()
            pan = pan.cuda()
            target = target.cuda()
        output = model(pan,mul)
        loss = criterion(output, target)
        psnr=criterion_psnr(output, target)
        ssim = criterion_ssim(output, target)
        sam = criterion_sam(output, target)
        # accumulation training
        if accumulation_steps-1:
            loss=loss/accumulation_steps
            loss.backward()
            if ((iteration + 1) % accumulation_steps) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),opt.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        endtime = time.time()
        temp=loss.data
        temp=temp.cpu()
        loss_vaule.append(temp.numpy().astype(np.float32))
        # print("\r ===> Epoch[{}]({}/{}): Loss:{:.10f} SSIM:{:.4f} PSNR:{:.4f} SAM:{:.4f} Time:{:.6f} lr={}".format(epoch,iteration,len(training_data_loader), loss.data,ssim.data,psnr,sam.data,endtime - starttime,optimizer.param_groups[0]["lr"]),end='')


def save_bestchkpt(model, epoch,file):
    model_out_path = '%s%s' % (file, "Best_model.pth")
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(file):
        os.makedirs(file)
    torch.save(state, model_out_path)
    print("\r Checkpoint saved to {}".format(model_out_path),end='')

def save_checkpoint(model, epoch,file):
    model_out_path = '%s%s%d%s' % (file, "model_epoch_",epoch,".pth")
    state = {"epoch": epoch ,"model": model,'optimizer_state_dict': optimizer.state_dict(),'scheduler_state_dict': scheduler.state_dict(),}
    if not os.path.exists(file):
        os.makedirs(file)
    torch.save(state, model_out_path)
    print("\r Checkpoint saved to {}".format(model_out_path),end='')

def validate(val_loader, model,epoch):
    with torch.no_grad():
        model.eval()

        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        losses_ssim = AverageMeter()
        losses_sam = AverageMeter()
        losses_ergas = AverageMeter()
        for i,batch in enumerate(val_loader,0):
            mul, pan, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])

            if opt.cuda:
                mul = mul.cuda()
                pan = pan.cuda()
                target = target.cuda()
            # compute output
            output = model(pan,mul)
            if epoch%20==0:
                imagepath=outfile+'/'+str(epoch)+'/'
                if not os.path.exists(imagepath):
                    os.makedirs(imagepath)
                torch2png(mul[0, [14, 7, 2], :, :], imagepath+'Img'+str(i)+'hs.png')
                torch2png(pan[0, [2, 1, 0], :, :], imagepath+'Img'+str(i)+'ms.png')
                torch2png(output[0, [14, 7, 2], :, :], imagepath+'Img'+str(i)+'output.png')
                torch2png(target[0, [14, 7, 2], :, :], imagepath + 'Img' + str(i) + 'gt.png')
                torch2tiff(mul[0, :, :, :], imagepath+'Tiff'+str(i)+'hs.tiff')
                torch2tiff(pan[0, :, :, :], imagepath+'Tiff'+str(i)+'ms.tiff')
                torch2tiff(output[0, :, :, :], imagepath+'Tiff'+str(i)+'output.tiff')
                torch2tiff(target[0, :, :, :], imagepath + 'Tiff' + str(i) + 'gt.tiff')
                writer.add_image('testimg/HSI'+str(i), mul[0, [14, 7, 2], :, :], epoch,dataformats='CHW')
                writer.add_image('testimg/MSI' + str(i), pan[0, [2, 1, 0], :, :], epoch,dataformats='CHW')
                writer.add_image('testimg/Out' + str(i), output[0, [14, 7, 2], :, :], epoch,dataformats='CHW')
                writer.add_image('testimg/Targ' + str(i), target[0, [14, 7, 2], :, :], epoch,dataformats='CHW')

            loss_rmse = criterion_cc(output, target)
            loss_psnr = criterion_psnr(output, target)
            loss_ssim = criterion_ssim(output, target)
            loss_sam = criterion_sam(output, target)
            loss_ergas = criterion_ergas(output, target,ratio)
            # record loss
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            losses_ssim.update(loss_ssim.data)
            losses_sam.update(loss_sam.data)
            losses_ergas.update(loss_ergas.data)
    return losses_rmse.avg, losses_psnr.avg,losses_ssim.avg, losses_sam.avg, losses_ergas.avg



print("===> Setting GPU")
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

    # optionally resume from a checkpoint
if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

print("===> Training")
rmse = []
psnr = []
ssim = []
sam = []
ergas = []
Best_psnr=0

for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    loss = []
    time_sta = time.time()
    train(training_data_loader, optimizer, model, criterion, epoch, opt.accumulation_steps, scheduler,loss)
    writer.add_scalar('train/loss', np.mean(loss), epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], epoch)

    if epoch % 5 == 0:
        save_checkpoint(model, epoch, filepath)
    time_end = time.time()
    print("{}===> Epoch[{}]: Loss: {:.6f} Time:{:.4f} ".format(filepath[filepath.index('/'):], epoch, np.mean(loss),
                                                               time_end - time_sta))

    logger_train.info(
        "===> Epoch[{}]: Loss: {:.6f} Time:{:.4f}  lr={}".format(epoch, np.mean(loss), time_end - time_sta,
                                                                 optimizer.param_groups[0]["lr"]))

    time_sta_test = time.time()
    rmse_loss, psnr_loss, ssim_loss, sam_loss,ergas_loss =validate(test_data_loader, model,epoch)
    rmse.append(rmse_loss)
    psnr.append(psnr_loss)
    ssim.append(ssim_loss)
    sam.append(sam_loss)
    ergas.append(ergas_loss)
    if  psnr_loss>Best_psnr:
        save_bestchkpt(model, epoch, filepath)
        Best_psnr=psnr_loss
    time_end_test = time.time()
    writer.add_scalar('test/rmse', rmse_loss, epoch)
    writer.add_scalar('test/psnr', psnr_loss, epoch)
    writer.add_scalar('test/ssim', ssim_loss, epoch)
    writer.add_scalar('test/sam', sam_loss, epoch)
    writer.add_scalar('test/ergas', ergas_loss, epoch)
    print("     Testing: CC {:.6f}, PSNR {:.6f}, SSIM {:.6f}, SAM {:.6f}, ERGAS {:.6f}   Time:{:.6f}".format(rmse_loss, psnr_loss, ssim_loss, sam_loss,ergas_loss,time_end_test-time_sta_test))
    logger_test.info("Epoch[{}]: CC {:.6f}, PSNR {:.6f}, SSIM {:.6f}, SAM {:.6f}, ERGAS {:.6f}   Time:{:.6f}".format(epoch,rmse_loss, psnr_loss, ssim_loss, sam_loss,ergas_loss,time_end_test-time_sta_test))
bestepoch = psnr.index(max(psnr))
logger_test.info(
    "Best Epoch[{}]: CC {:.6f}, PSNR {:.6f}, SSIM {:.6f}, SAM {:.6f}, ERGAS {:.6f}".format(bestepoch, rmse[bestepoch],
                                                                                           psnr[bestepoch],
                                                                                           ssim[bestepoch],
                                                                                           sam[bestepoch],
                                                                                           ergas[bestepoch]))
logger_test.info(str(opt))
writer.close()

import os
import glob

path = '%s%s' % (filepath, 'model_epoch_*.pth')
files_to_delete = glob.glob(path)

for file in files_to_delete:
    try:
        os.remove(file)
    except FileNotFoundError:
        print(f"{file} not found or already deleted.")




