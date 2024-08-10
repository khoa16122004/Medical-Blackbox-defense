import torch
import torch.nn as nn





class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = torch.load("/mlcv2/WorkingSpace/Personal/baotg/brain/Black-Box-Defense/cass-r50-isic.pt", 
                                  map_location=torch.device("cuda:0"))
    
    def forward(self, x):
        x = self.encoder(x)
        return x.unsqueeze(-1).unsqueeze(-1)
         
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1000, 512, kernel_size=24),  # 512x24x24
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256x48x48
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128x96x96
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x192x192
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 3x384x384
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        return self.decoder(x)

# Khởi tạo encoder và decoder
encoder = Encoder().cuda()
decoder = Decoder().cuda()

# Ví dụ về cách sử dụng encoder và decoder
input_image = torch.randn((1, 3, 384, 384)).cuda()  # Batch size là 1
encoded = encoder(input_image)
print(encoded.shape)
decoded = decoder(encoded)

print(f"Input shape: {input_image.shape}")
print(f"Encoded shape: {encoded.shape}")
print(f"Decoded shape: {decoded.shape}")
