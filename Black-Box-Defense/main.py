from architectures import get_architecture
from datasets import get_dataset, DATASETS
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer, Adam
import itertools
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.nn import MSELoss, CrossEntropyLoss
from train_utils import AverageMeter, accuracy, init_logfile, log, copy_code, requires_grad_, measurement
from torchvision.utils import save_image
import argparse
import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from config import *

parser = argparse.ArgumentParser(description="Image Recontruction")

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--dataset', type=str, choices=DATASETS, default="Brain_Tumor")
parser.add_argument('--batch', default=32 , type=int)

parser.add_argument('--encoder_arch', type=str, default="Encoder_Vit_1000")
parser.add_argument('--decoder_arch', type=str, default="Decoder_Vit_1000")
parser.add_argument('--pretrained_encoder', type=str, default=None)
parser.add_argument('--pretrained_decoder', type=str, default=None)
parser.add_argument('--out_dir', type=str, default="Brain_Recontruction_Vit_1000")
parser.add_argument('--pretrained_denoiser', default='', type=str, help='path to a pretrained denoiser')
parser.add_argument('--noise_sd', default=0.25, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step_size', type=int, default=100,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--img_path', type=str, default=None)
args = parser.parse_args()


def train_cae(encoder, 
              decoder, 
              train_loader, 
              val_loader, 
              criterion, 
              optimizer, 
              scheduler, 
              logfilename):
    
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    best_score = None
    saving = False
    
    for epoch in tqdm(range(args.epochs)):
        encoder.train()
        decoder.train()
        
        saving = False  

        for (imgs, labels) in tqdm(train_loader):
            imgs = imgs.cuda()
            encoder_imgs = encoder(imgs)
            decoder_imgs = decoder(encoder_imgs)
            loss = criterion(decoder_imgs, imgs)
            train_loss_meter.update(loss, imgs.shape[0])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 
        
        scheduler.step(epoch)
        args.lr = scheduler.get_lr()[0]
        
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            for (imgs, labels) in val_loader:
                imgs = imgs.cuda()
                encoder_imgs = encoder(imgs)
                decoder_imgs = decoder(encoder_imgs)
                loss = criterion(decoder_imgs, imgs)
                val_loss_meter.update(loss, imgs.shape[0])
            
            if not best_score or val_loss_meter.avg < best_score:
                best_score = val_loss_meter.avg
                torch.save(encoder, os.path.join(args.out_dir,"best_encoder.pth"))
                torch.save(decoder, os.path.join(args.out_dir,"best_decoder.pth"))
                saving = True
            
            
        log(logfilename, f"{epoch}: {train_loss_meter.avg}  {val_loss_meter.avg} {saving}")
        train_loss_meter.reset()
        val_loss_meter.reset()

def test_cae(encoder, decoder, test_loader, criterion, logfilename):
    test_loss_meter = AverageMeter()
    
    encoder.eval()
    decoder.eval() 
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader):
            imgs = imgs.cuda()
            encoder_imgs = encoder(imgs)
            decoder_imgs = decoder(encoder_imgs)
            loss = criterion(decoder_imgs, imgs)
            test_loss_meter.update(loss, imgs.shape[0])
    
    log(logfilename, f"Test_loss: {test_loss_meter.avg}")
    
def inference(encoder, decoder, img_path, 
              transform = transforms.Compose([transforms.Resize((384,384)), 
                                  transforms.ToTensor()])):
    encoder.eval()
    decoder.eval() 
    with torch.no_grad():
        img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
        img = img.cuda()
        encoder_img = encoder(img)
        decoder_img = decoder(encoder_img)
        save_image(decoder_img, os.path.join(args.out_dir, "recon.jpg"))
    
def inference_classify(encoder, decoder, model, test_loader):
    encoder.eval()
    decoder.eval()
    
    acc_meter = AverageMeter()
    acc_ = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            imgs, labels = data
            imgs = imgs.cuda()
            labels = labels.cuda()
            output = model(decoder(encoder(imgs)))
            # if torch.argmax(output) == labels:
            #     acc_ += 1
            acc = accuracy(output, labels)
            # if acc != 1:
            #     print(labels)
            
            acc_meter.update(acc[0].item(), imgs.shape[0])

    print(f"Acc: {acc_meter.avg:.4f}")

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    logfilename = os.path.join(args.out_dir, 'log.txt')
    if not os.path.isfile(logfilename):
        init_logfile(logfilename, "epoch\Train_Loss\Val_Loss")

        
    # dataset 
    train_dataset = get_dataset(args.dataset, 'Train')
    val_dataset = get_dataset(args.dataset, 'Val')
    test_dataset = get_dataset(args.dataset, 'Test')
    train_loader = DataLoader(train_dataset, args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch, shuffle=False)


    # conv autoencoder
    encoder = get_architecture(args.encoder_arch, args.dataset).cuda()
    decoder = get_architecture(args.decoder_arch, args.dataset).cuda()

    
    
    if args.pretrained_encoder:
        encoder = torch.load(args.pretrained_encoder)
    if args.pretrained_decoder:
        decoder = torch.load(args.pretrained_decoder)

    # optimizer
    optimizer = Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    
    # print(decoder(encoder(torch.randn(1, 3, 384, 384).cuda())).shape)
    # criterion
    criterion = MSELoss(size_average=None, reduce=None, reduction='mean').cuda()

    if args.mode == "train":
        train_cae(encoder, decoder, train_loader, val_loader, criterion, optimizer, scheduler, logfilename)
    elif args.mode == "test":
        test_cae(encoder, decoder, test_loader, criterion, logfilename)
    
    elif args.mode == "inference_classifier":
        model = torch.load(checkpoint_path, map_location=torch.device(device)).eval().cuda()
        inference_classify(encoder, decoder, model, test_loader)
    
    else:
        # print(decoder(encoder(torch.randn(3, 384, 384).cuda())).shape)
        inference(encoder, decoder, args.img_path)
            
if __name__ == "__main__":
    main()
    
