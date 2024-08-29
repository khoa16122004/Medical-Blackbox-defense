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

parser.add_argument('--mode', type=str, default='infer_DS') # ["CLF", 'AE_DS', 'DS', 'infer_AE_DS', 'infer_DS']
parser.add_argument('--dataset', type=str, choices=DATASETS, default="Brain_Tumor")
parser.add_argument('--batch', default=32 , type=int)
parser.add_argument('--classifier', default='vit_sipadmek', type=str,
                    help='path to the classifier used with the `classificaiton`'
                         'or `stability` objectives of the denoiser.')
parser.add_argument('--encoder_arch', type=None, default="Encoder_Vit_1000")
parser.add_argument('--pretrained_denoiser', default=None, type=str, help='path to a pretrained denoiser')
parser.add_argument('--noise_sd', default=0.25, type=float,
                    help="standard deviation of noise distribution for data augmentation")
parser.add_argument('--arch', type=str, default="cifar_dncnn_wide")

parser.add_argument('--decoder_arch', type=None, default="Decoder_Vit_1000")
parser.add_argument('--pretrained_encoder', type=str, default=None)
parser.add_argument('--pretrained_decoder', type=str, default=None)
parser.add_argument('--out_dir', type=str, default="experiment\SIPADMEK_CE_0.25")
parser.add_argument('--img_path', type=str, default=None)



args = parser.parse_args()


def classfier(model, test_loader):
    acc_meter = AverageMeter()
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            imgs = model(imgs)
            acc = accuracy(imgs, labels)
            acc_meter.update(acc[0].item(), imgs.shape[0])
    print(f"Acc: {acc_meter.avg}")

def DS(model, denoiser, encoder, decoder, test_loader, mode, logfilename):    
    acc_meter = AverageMeter()
    
    model.eval()
    denoiser.eval()
    
    if encoder or decoder:
        encoder.eval()
        decoder.eval() 
    
    with torch.no_grad():
        for (imgs, labels) in tqdm(test_loader):
            imgs, labels = imgs.cuda(), labels.cuda()
            imgs = imgs + torch.randn_like(imgs, device='cuda') * args.noise_sd
            imgs = denoiser(imgs)
            if "AE_DS" in mode:
                imgs = encoder(imgs)
                imgs = decoder(imgs)
                
            imgs = model(imgs)
            acc = accuracy(imgs, labels)
            acc_meter.update(acc[0].item(), imgs.shape[0])
    
    log(logfilename, f"Acc_loss: {acc_meter.avg}")
    
def inference(test_dataset, index, denoiser, encoder, decoder, img_path, mode, 
              transform = transforms.Compose([transforms.Resize((384,384)), 
                                  transforms.ToTensor()])):
    
    denoiser.eval()
    if encoder or decoder:
        encoder.eval()
        decoder.eval() 
    
    with torch.no_grad():
        if img_path:
            img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
        else:
            img = test_dataset[index][0].unsqueeze(0)
        save_image(img, os.path.join(args.out_dir, "original.jpg"))

        img = img.cuda()
        img = img + torch.randn_like(img, device='cuda') * args.noise_sd
        save_image(img, os.path.join(args.out_dir, "noise.jpg"))

        img = denoiser(img)
        save_image(img, os.path.join(args.out_dir, "ds.jpg"))

        if "AE_DS" in mode:
            img = encoder(img)
            img = decoder(encoder_img)
            save_image(img, os.path.join(args.out_dir, "ae_ds.jpg"))
    
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    logfilename = os.path.join(args.out_dir, 'log.txt')
    if not os.path.isfile(logfilename):
        init_logfile(logfilename, "epoch\Train_Loss\Val_Loss")

        
    # dataset 
    if args.dataset == "mnist":
        test_dataset = get_dataset(args.dataset, 'test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)
    else:
        test_dataset = get_dataset(args.dataset, 'Test')
        test_loader = DataLoader(test_dataset, args.batch, shuffle=False)

    # encoder
    encoder = None
    decoder = None
    if args.pretrained_encoder:
        encoder = torch.load(args.pretrained_encoder)
    if args.pretrained_decoder:
        decoder = torch.load(args.pretrained_decoder)
    
    # denosier
    if args.pretrained_denoiser:
        checkpoint = torch.load(args.pretrained_denoiser)
        denoiser = get_architecture(checkpoint['arch'], args.dataset)
        print(denoiser.load_state_dict(checkpoint['state_dict']))

    # classifier
    print(args.classifier)
    model = get_architecture(args.classifier, args.dataset)
    # checkpoint = torch.load(args.classifier)
    # print(checkpoint['arch'])
    # model = get_architecture(checkpoint['arch'], args.dataset)
    # model.load_state_dict(checkpoint['state_dict'])

    if args.mode == "clf":
        classfier(model, test_loader)
    
    elif 'infer' in args.mode:
       inference(test_dataset, 200, denoiser, encoder, decoder, args.img_path, args.mode)
    else:
       DS(model, denoiser, encoder, decoder, test_loader, args.mode, logfilename)
            
if __name__ == "__main__":
    main()
    
