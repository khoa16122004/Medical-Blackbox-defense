import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD, Adam
from utils import *
from tqdm import tqdm
from config import *
from dataset import BrainTumorDataset
from architech import *
import logging

# Uncomment and use argparse if needed
# import argparse
# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
#                     help='initial learning rate', dest='lr')
# parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log)')
# parser.add_argument('--epochs', default=10, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--arch', type=str, choices=CLASSIFIERS_ARCHITECTURES)
# parser.add_argument('--dataset', type=str, choices=DATASETS)
# parser.add_argument('--optimizer', default='Adam', type=str,
#                     help='SGD, Adam, or Adam then SGD', choices=['SGD', 'Adam', 'AdamThenSGD'])
# parser.add_argument('--gpu', default=None, type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')

# args = parser.parse_args()

# Use the config or argparse arguments to set these variables
# lr = args.lr
# epochs = args.epochs
# device = args.gpu if args.gpu else 'cuda:0'
# OUTDIR_TRAIN = args.outdir

os.environ['CUDA_VISIBLE_DEVICES'] = device

if not os.path.exists(OUTDIR_TRAIN):
    os.mkdir(OUTDIR_TRAIN)

# Set up logging
logging.basicConfig(filename=f'{OUTDIR_TRAIN}/training_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Starting training process')

train_dataset = BrainTumorDataset(ARGUMENT_PATH, ARGUMENT_DIR)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# model = Classification().train().cuda()
# model = VGG("VGG19").train().cuda()
model = ResNet50().train().cuda()
criterion = nn.CrossEntropyLoss(size_average=None, reduce=None, reduction='mean').cuda()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Optionally, add a learning rate scheduler
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

acc_meter = AverageMeter()
losses_meter = AverageMeter()

best_acc = None

for epoch in tqdm(range(epochs)):
    losses_meter.reset()
    acc_meter.reset()
    for data in tqdm(train_loader):
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        output = model(imgs)
        loss = criterion(output, labels)
        print(loss)
        acc = accuracy(output, labels)

        losses_meter.update(loss.item(), imgs.shape[0])
        acc_meter.update(acc[0].item(), imgs.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()  # Step the scheduler at the end of the epoch if using

    # Log epoch metrics
    logging.info(f"Epoch {epoch} Loss: {losses_meter.avg:.4f} Acc: {acc_meter.avg:.4f}")
    print(f"Epoch {epoch} Loss: {losses_meter.avg:.4f} Acc: {acc_meter.avg:.4f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"{OUTDIR_TRAIN}/ep{epoch}.pth")

    if not best_acc or acc_meter.avg > best_acc:
        best_acc = acc_meter.avg
        torch.save(model.state_dict(), f"{OUTDIR_TRAIN}/best.pth")