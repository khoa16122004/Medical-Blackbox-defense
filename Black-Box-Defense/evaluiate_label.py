import torch
from torch import nn
from torchvision import transforms
from architectures import get_architecture
from PIL import Image

label_str2num = {'glioma': '0',
                    'meningioma': '1',
                    'notumor': '2',
                    'pituitary': '3',
                    }

dataset_name = "Brain_Tumor"
classifier_name = "cass_classifier"
img_path = r"Dataset\Brain_Tumor\Testing\pituitary\Te-pi_0010.jpg"
fgsm_path = r"Dataset\Brain_Tumor\AT_FGSM\3\Te-pi_0010.jpg"
pgd_path = r"Dataset\Brain_Tumor\AT_PGD\3\Te-pi_0010.jpg"
ddn_path = r"Dataset\Brain_Tumor\AT_DDN\3\Te-pi_0010.jpg"
# fgsm_path = img_path.replace("Testing", "AT_FGSM")
# pgd_path = img_path.replace("Testing", "AT_PGD")
# ddn_path = img_path.replace("Testing", "AT_DDN")
print(img_path)
print(fgsm_path)
print(pgd_path)
print(ddn_path)

imgs = [fgsm_path, pgd_path, ddn_path]

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

model = get_architecture(classifier_name, dataset_name).cuda().eval()

for img_path in imgs:
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    output = model(img.unsqueeze(0).cuda())
    pred_label = output.argmax(dim=1)
    print(pred_label)