import torch
import torch.nn as nn
from adversarial_library.adv_lib.attacks import fast_gradient_sign_method, ddn
from datasets import get_dataset
from torch.utils.data import DataLoader

from torch.nn import MSELoss, CrossEntropyLoss
import os
from torchvision.utils import save_image
from tqdm import tqdm
from torch import Tensor
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--classifier', default='classification', type=str)
parser.add_argument('--dataset', type=str, default="Brain_Tumor")
parser.add_argument('--type', type=str, default="PGD", choices=["PGD", "FGSM", "DDN"])


args = parser.parse_args()


class Model_(nn.Module):
    def __init__(self, model):
        super(Model_, self).__init__()
        self.main = model
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.main(x)
        x = self.softmax(x)
        return x

def create_adversarial(model: nn.Module, 
                       criterion: nn.Module, 
                       img: Tensor,
                       label: Tensor,
                       type: str):
    
    
    
    
    
    if type == "PGD":
        adversarial_image = fast_gradient_sign_method.PGD(model,
                                                          img,
                                                          criterion,
                                                          label)

    elif type == "FGSM":
        adversarial_image = fast_gradient_sign_method.FGSM(model, img, criterion, label)
    elif type == "DDN":
        adversarial_image = ddn(model=model, inputs=img, labels=label, steps=100)

    return adversarial_image
        

class Model_(nn.Module):
    def __init__(self, model):
        super(Model_, self).__init__()
        self.main = model
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.main(x)
        x = self.softmax(x)
        return x


def main():
    test_dataset = get_dataset(args.dataset, "Test")
    test_loader = DataLoader(test_dataset, 1, shuffle=False)


    model = torch.load(args.classifier, map_location="cuda:0")
    if args.type == "DDN":        
        model = Model_(model).train().cuda()
    criterion = CrossEntropyLoss(size_average=None, reduce=False, reduction='none').cuda()
    
    if args.dataset == "Brain_Tumor":
        brain_tumor_dir = "Dataset/Brain_Tumor"
        out_dir = os.path.join(brain_tumor_dir, f"AT_{args.type}")
        os.makedirs(out_dir, exist_ok=True)
    elif args.dataset == "SIPADMEK":
        SIPADMEK = "Dataset/SIPADMEK"
        out_dir = os.path.join(SIPADMEK, f"AT_{args.type}")
        os.makedirs(out_dir, exist_ok=True)
        
            
    for i, (img, label, img_basename) in tqdm(enumerate((test_loader))):
        img, label = img.cuda(), label.cuda()
        label_dir = os.path.join(out_dir, str(label.item()))
        os.makedirs(label_dir, exist_ok=True)

        ad = create_adversarial(model, 
                                criterion, 
                                img,
                                label,
                                args.type)
        out_path = os.path.join(label_dir, img_basename[0])
        save_image(ad, out_path)
    
if __name__ == "__main__":
    main()

        
        
        

        
        
        
    
