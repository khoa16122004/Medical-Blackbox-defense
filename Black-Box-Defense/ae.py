import torch
import torch.nn as nn
from architectures import get_architecture


encoder = get_architecture(arch="Sadnet_Encoder", dataset="SIPADMEK")
decoder = get_architecture(arch="Sadnet_Decoder", dataset="SIPADMEK")

ins = torch.rand(1, 3, 384, 384).cuda()
ins = encoder(ins)
print(ins[0].shape)
ins = decoder(ins[0].cuda(), ins[1])
print(ins.shape)
