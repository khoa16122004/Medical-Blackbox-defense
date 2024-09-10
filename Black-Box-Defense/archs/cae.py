
# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])
#
# dataset = MNIST('./data', transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



import torch
import torch.nn as nn



import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from torchvision.ops import deform_conv2d
import math
import logging
logger = logging.getLogger('base')


class ModulatedDeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 deformable_groups=1,
                 extra_offset_mask=True,
                 offset_in_channel=32
                 ):
        super(ModulatedDeformableConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.extra_offset_mask = extra_offset_mask

        self.conv_offset_mask = nn.Conv2d(offset_in_channel,
                                     deformable_groups * 3 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        self.init_offset()

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.init_weights()

    def init_weights(self):
        n = self.in_channels * self.kernel_size * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        torch.nn.init.constant_(self.conv_offset_mask.weight, 0.)
        torch.nn.init.constant_(self.conv_offset_mask.bias, 0.)


    def forward(self, x):
        if self.extra_offset_mask:
            # x = [input, features]
            offset_mask = self.conv_offset_mask(x[1])
            x = x[0]

        else:
            offset_mask = self.conv_offset_mask(x)

            
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_mean = torch.mean(torch.abs(offset))
        # if offset_mean > max(x.shape[2:]):
        #     print(x.shape)
        #     logger.warning('Offset mean is {}, larger than max(h, w).'.format(offset_mean))

        out = deform_conv2d(input=x,
                            offset=offset,
                            weight=self.weight,
                            bias=self.bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            mask=mask
                            )


        return out

#==============================================================================#
class ResBlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.conv2(conv1)
        out = x + conv2
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class RSABlock(nn.Module):

    def __init__(self, input_channel=32, output_channel=32, offset_channel=32):
        super().__init__()
        self.in_channel = input_channel
        self.out_channel = output_channel
        if self.in_channel != self.out_channel:
            self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        self.dcnpack = ModulatedDeformableConv2d(output_channel, output_channel, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                            extra_offset_mask=True, offset_in_channel=offset_channel)
        self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, offset):
        if self.in_channel != self.out_channel:
            x = self.conv0(x)
        fea = self.lrelu(self.dcnpack([x, offset]))
        out = self.conv1(fea) + x
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class OffsetBlock(nn.Module):

    def __init__(self, input_channel=32, offset_channel=32, last_offset=False):
        super().__init__()
        self.offset_conv1 = nn.Conv2d(input_channel, offset_channel, 3, 1, 1)  # concat for diff
        if last_offset:
            self.offset_conv2 = nn.Conv2d(offset_channel*2, offset_channel, 3, 1, 1)  # concat for offset
        self.offset_conv3 = nn.Conv2d(offset_channel, offset_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x, last_offset=None):
        offset = self.lrelu(self.offset_conv1(x))
        if last_offset is not None:
            last_offset = F.interpolate(last_offset, scale_factor=2, mode='bilinear', align_corners=False)
            offset = self.lrelu(self.offset_conv2(torch.cat([offset, last_offset * 2], dim=1)))
        offset = self.lrelu(self.offset_conv3(offset))
        return offset
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

class ContextBlock(nn.Module):
    def __init__(self, input_channel=32, output_channel=32, square=False):
        super().__init__()
        self.conv0 = nn.Conv2d(input_channel, output_channel, 1, 1)
        if square:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 8, 8)
        else:
            self.conv1 = nn.Conv2d(output_channel, output_channel, 3, 1, 1, 1)
            self.conv2 = nn.Conv2d(output_channel, output_channel, 3, 1, 2, 2)
            self.conv3 = nn.Conv2d(output_channel, output_channel, 3, 1, 3, 3)
            self.conv4 = nn.Conv2d(output_channel, output_channel, 3, 1, 4, 4)
        self.fusion = nn.Conv2d(4*output_channel, input_channel, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.initialize_weights()

    def forward(self, x):
        x_reduce = self.conv0(x)
        conv1 = self.lrelu(self.conv1(x_reduce))
        conv2 = self.lrelu(self.conv2(x_reduce))
        conv3 = self.lrelu(self.conv3(x_reduce))
        conv4 = self.lrelu(self.conv4(x_reduce))
        out = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.fusion(out) + x
        return out
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


#===============================================================================#

class Sadnet_Encoder(nn.Module):

    def __init__(self, input_channel=3, n_channel=32):
        super().__init__()

        self.res1 = ResBlock(input_channel, n_channel)
        self.down1 = nn.Conv2d(n_channel, n_channel * 2, 2, 2)
        self.res2 = ResBlock(n_channel * 2, n_channel * 2)
        self.down2 = nn.Conv2d(n_channel * 2, n_channel * 4, 2, 2)
        self.res3 = ResBlock(n_channel * 4, n_channel * 4)
        self.down3 = nn.Conv2d(n_channel * 4, n_channel * 8, 2, 2)
        self.res4 = ResBlock(n_channel * 8, n_channel * 8)
        self.down4 = nn.Conv2d(n_channel * 8, n_channel * 8, 2, 2)  # (256, 24, 24)
        self.res5 = ResBlock(n_channel * 8, n_channel * 8)
        self.down5 = nn.Conv2d(n_channel * 8, n_channel * 8, 2, 2)  # (256, 12, 12)
        self.res6 = ResBlock(n_channel * 8, n_channel * 8)

        # Down channel
        self.down_c = nn.Conv2d(n_channel * 8, 8, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        conv1 = self.res1(x)
        pool1 = self.lrelu(self.down1(conv1))
        conv2 = self.res2(pool1)
        pool2 = self.lrelu(self.down2(conv2))
        conv3 = self.res3(pool2)
        pool3 = self.lrelu(self.down3(conv3))
        conv4 = self.res4(pool3)
        pool4 = self.lrelu(self.down4(conv4))  # (256, 24, 24)
        conv5 = self.res5(pool4)
        pool5 = self.lrelu(self.down5(conv5))  # (256, 12, 12)
        conv6 = self.res6(pool5)
        down_c = self.down_c(conv6)  # (8, 12, 12)

        return down_c, [conv1, conv2, conv3, conv4, conv5, conv6]


class Sadnet_Decoder(nn.Module):

    def __init__(self, output_channel=3, n_channel=32, offset_channel=32):
        super().__init__()

        # Up channel
        self.up_c = nn.Conv2d(8, n_channel * 8, 1)

        self.context = ContextBlock(n_channel * 8, n_channel * 8, square=False)
        self.offset6 = OffsetBlock(n_channel * 8, offset_channel, False)
        self.dres6 = RSABlock(n_channel * 8, n_channel * 8, offset_channel)

        self.up5 = nn.ConvTranspose2d(n_channel * 8, n_channel * 8, 2, 2)
        self.dconv5_1 = nn.Conv2d(n_channel * 16, n_channel * 8, 1, 1)
        self.offset5 = OffsetBlock(n_channel * 8, offset_channel, True)
        self.dres5 = RSABlock(n_channel * 8, n_channel * 8, offset_channel)

        self.up4 = nn.ConvTranspose2d(n_channel * 8, n_channel * 8, 2, 2)
        self.dconv4_1 = nn.Conv2d(n_channel * 16, n_channel * 8, 1, 1)
        self.offset4 = OffsetBlock(n_channel * 8, offset_channel, True)
        self.dres4 = RSABlock(n_channel * 8, n_channel * 8, offset_channel)

        self.up3 = nn.ConvTranspose2d(n_channel * 8, n_channel * 4, 2, 2)
        self.dconv3_1 = nn.Conv2d(n_channel * 8, n_channel * 4, 1, 1)
        self.offset3 = OffsetBlock(n_channel * 4, offset_channel, True)
        self.dres3 = RSABlock(n_channel * 4, n_channel * 4, offset_channel)

        self.up2 = nn.ConvTranspose2d(n_channel * 4, n_channel * 2, 2, 2)
        self.dconv2_1 = nn.Conv2d(n_channel * 4, n_channel * 2, 1, 1)
        self.offset2 = OffsetBlock(n_channel * 2, offset_channel, True)
        self.dres2 = RSABlock(n_channel * 2, n_channel * 2, offset_channel)

        self.up1 = nn.ConvTranspose2d(n_channel * 2, n_channel, 2, 2)
        self.dconv1_1 = nn.Conv2d(n_channel * 2, n_channel, 1, 1)
        self.offset1 = OffsetBlock(n_channel, offset_channel, True)
        self.dres1 = RSABlock(n_channel, n_channel, offset_channel)

        self.out = nn.Conv2d(n_channel, output_channel, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, down_c, encoder_features):
        conv1, conv2, conv3, conv4, conv5, conv6 = encoder_features

        up_c = self.up_c(down_c) + conv6  # (256, 12, 12)
        conv6_back = self.context(up_c)

        L6_offset = self.offset6(conv6_back, None)
        dconv6 = self.dres6(conv6_back, L6_offset)

        up5 = torch.cat([self.up5(dconv6), conv5], 1)  # (512, 24, 24)
        up5 = self.dconv5_1(up5)
        L5_offset = self.offset5(up5, L6_offset)
        dconv5 = self.dres5(up5, L5_offset)

        up4 = torch.cat([self.up4(dconv5), conv4], 1)  # (512, 48, 48)
        up4 = self.dconv4_1(up4)
        L4_offset = self.offset4(up4, L5_offset)
        dconv4 = self.dres4(up4, L4_offset)

        up3 = torch.cat([self.up3(dconv4), conv3], 1)
        up3 = self.dconv3_1(up3)
        L3_offset = self.offset3(up3, L4_offset)
        dconv3 = self.dres3(up3, L3_offset)

        up2 = torch.cat([self.up2(dconv3), conv2], 1)
        up2 = self.dconv2_1(up2)
        L2_offset = self.offset2(up2, L3_offset)
        dconv2 = self.dres2(up2, L2_offset)

        up1 = torch.cat([self.up1(dconv2), conv1], 1)
        up1 = self.dconv1_1(up1)
        L1_offset = self.offset1(up1, L2_offset)
        dconv1 = self.dres1(up1, L1_offset)

        out = self.out(dconv1)
        return out


        



class Encoder_Vit_1000(nn.Module):
    def __init__(self):
        super(Encoder_Vit_1000, self).__init__()
        self.encoder = torch.load("Black-Box-Defense/cass-r50-isic.pt", 
                                  map_location=torch.device("cuda:0"))
    
    def forward(self, x):
        x = self.encoder(x)
        return x.unsqueeze(-1).unsqueeze(-1)


class Decoder_Vit_1000(nn.Module):
    def __init__(self):
        super(Decoder_Vit_1000, self).__init__()
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
            nn.Sigmoid()  # Áp dụng sigmoid để đảm bảo đầu ra trong khoảng [0, 1]
        )
    
    def forward(self, x):
        return self.decoder(x)
class Encoder_1000(nn.Module):
    def __init__(self):
        super(Encoder_1000, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 64x192x192
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128x96x96
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256x48x48
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512x24x24
            nn.ReLU(True),
            nn.Conv2d(512, 1000, kernel_size=24)  # 1000x1x1
        )
    
    def forward(self, x):
        return self.encoder(x)
class Decoder_1000(nn.Module):
    def __init__(self):
        super(Decoder_1000, self).__init__()
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

class Encoder_768(nn.Module):
    def __init__(self):
        super(Encoder_768, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [32, 192, 192]
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [64, 96, 96]
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [128, 48, 48]
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # [256, 24, 24]
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # [512, 12, 12]
            nn.ReLU(True),
            nn.Conv2d(512, 768, kernel_size=12),  # [768, 1, 1]
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder_768(nn.Module):
    def __init__(self):
        super(Decoder_768, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, kernel_size=12),  # [512, 12, 12]
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # [256, 24, 24]
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # [128, 48, 48]
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [64, 96, 96]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [32, 192, 192]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # [3, 384, 384]
            nn.Sigmoid()  # Optional: to ensure output is in range [0, 1]
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class Custom_Encoder_256(nn.Module):
    def __init__(self):
        super(Custom_Encoder_256, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), # Bx64x192x192
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Bx128x96x96
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Bx256x48x48
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # Bx512x24x24
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), # Bx1024x12x12
            nn.ReLU(True),
            nn.Conv2d(1024, 256, kernel_size=3, stride=2, padding=1), # Bx256x6x6
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=6), # Bx256x1x1
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.encoder(x)
class Custom_Decoder_256(nn.Module):
    def __init__(self):
        super(Custom_Decoder_256, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=6), # Bx256x6x6
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 1024, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx1024x12x12
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx512x24x24
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx256x48x48
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx128x96x96
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx64x192x192
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx3x384x384
            nn.ReLU() # Bx3x384x384
        )

    def forward(self, x):
        return self.decoder(x)




class Custom_Encoder(nn.Module):
    def __init__(self):
        super(Custom_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), # Bx64x192x192
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Bx128x96x96
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Bx256x48x48
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # Bx512x24x24
            nn.ReLU(True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), # Bx1024x12x12
            nn.ReLU(True),
            nn.Conv2d(1024, 192, kernel_size=3, stride=2, padding=1), # Bx192x6x6
            nn.ReLU(True),
            nn.Conv2d(192, 192, kernel_size=6), # Bx192x1x1
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.encoder(x)
class Custom_Decoder(nn.Module):
    def __init__(self):
        super(Custom_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 192, kernel_size=6), # Bx192x6x6
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 1024, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx1024x12x12
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx512x24x24
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx256x48x48
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx128x96x96
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx64x192x192
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # Bx3x384x384
            nn.Sigmoid() # Bx3x384x384
        )

    def forward(self, x):
        return self.decoder(x)







class MNIST_CAE(nn.Module):
    def __init__(self):
        super(MNIST_CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 28, 28
            nn.ReLU(True),

            nn.Conv2d(16, 32, 3, stride=3, padding=1), # b, 32, 10, 10
            nn.ReLU(True),

            nn.Conv2d(32, 16, 3, stride=3, padding=1), # b, 16, 4, 4
            nn.ReLU(True),

            nn.Conv2d(16, 8, 3, stride=3, padding=1),  # b, 8, 2, 2
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MNIST_Dim_Encoder(nn.Module):
    def __init__(self):
        super(MNIST_Dim_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, 4, stride=2, padding=1),      #(3, 14, 14)
            nn.ReLU(),
            nn.Conv2d(3, 12, 5, stride=3, padding=0),     #(12, 4, 4)
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class MNIST_Dim_Decoder(nn.Module):
    def __init__(self):
        super(MNIST_Dim_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12, 3, 5, stride=3, padding=0),   #(3, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, 4, stride=2, padding=1),   #(1, 28, 28)
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class CelebA_CAE(nn.Module):
    def __init__(self):
        super(CelebA_CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CIFAR_CAE(nn.Module):
    def __init__(self):
        super(CIFAR_CAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 48, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			#  nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class STL_Encoder(nn.Module):
    def __init__(self):
        super(STL_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 18, 5, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(18, 72, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(72, 144, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(144, 288, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(288, 576, 2, stride=2, padding=0),  # [batch, 576, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class STL_Decoder(nn.Module):
    def __init__(self):
        super(STL_Decoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(576, 288, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(288, 144, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(144, 72, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(72, 18, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(18, 3, 5, stride=3, padding=1),  # [batch, 12, 16, 16]
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_15552(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_15552, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 72, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(72, 144, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(144, 432, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(432, 864, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(864, 1728, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_15552(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_15552, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1728, 864, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(864, 432, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(432, 144, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(144, 72, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(72, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class ImageNet_Encoder_1152(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_1152, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(192, 384, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(384, 1152, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_1152(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_1152, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1152, 384, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96,48, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_1728(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_1728, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 36, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(36, 72, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(72, 144, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(144, 288, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(288, 576, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(576, 1728, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_1728(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_1728, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1728, 576, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(576, 288, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(288, 144, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(144,72, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(72, 36, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(36, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_2304(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_2304, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 48, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(192, 384, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(384, 768, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(768, 2304, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_2304(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_2304, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2304, 768, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(192,96, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(48, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class ImageNet_Encoder_3456(nn.Module):
    def __init__(self):
        super(ImageNet_Encoder_3456, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 48, 8, stride=6, padding=0),            # [batch, 18, 55, 55]
            nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=0),           # [batch, 72, 13, 13]
            nn.ReLU(),
            nn.Conv2d(96, 288, 3, stride=3, padding=0),           # [batch, 144, 6, 6]
            nn.ReLU(),
            nn.Conv2d(288, 576, 2, stride=2, padding=1),          # [batch, 288, 3, 3]
            nn.ReLU(),
            nn.Conv2d(576, 1152, 2, stride=2, padding=1),         # [batch, 864, 1, 1]
            nn.ReLU(),
            nn.Conv2d(1152, 3456, 3, stride=3, padding=0),        # [batch, 864, 1, 1]
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class ImageNet_Decoder_3456(nn.Module):
    def __init__(self):
        super(ImageNet_Decoder_3456, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3456, 1152, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(1152, 576, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(576, 288, 2, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(288,96, 3, stride=3, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 3, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(48, 3, 8, stride=6, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class TinyImageNet_Encoder(nn.Module):
    def __init__(self):
        super(TinyImageNet_Encoder, self).__init__()
        # Input size: [batch, 3, 64, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(6, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),

        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class TinyImageNet_Decoder(nn.Module):
    def __init__(self):
        super(TinyImageNet_Decoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 6, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 2, stride=2, padding= 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

class TinyImageNet_Encoder_768(nn.Module):
    def __init__(self):
        super(TinyImageNet_Encoder_768, self).__init__()
        # Input size: [batch, 3, 64, 64]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0), # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 2, stride=2, padding=0),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.Conv2d(24, 48, 4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),  # [batch, 96, 2, 2]
            nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),

        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class TinyImageNet_Decoder_768(nn.Module):
    def __init__(self):
        super(TinyImageNet_Decoder_768, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 2, stride=2, padding=0),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded



class Cifar_Encoder_48(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_48, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(24, 48, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_48(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_48, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(12, 6, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class Cifar_Encoder_96(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_96, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(48, 96, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_96(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_96, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class Cifar_Encoder_192_24(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_192_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_192_24(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_192_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded




class Cifar_Encoder_192(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_192, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
            nn.Conv2d(96, 192, 2, stride=2, padding=0),  # [batch, 192, 1, 1]
            nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 192, 1, 1]
        return encoded


class Cifar_Decoder_192(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_192, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 2, stride=2, padding=0),  # [batch, 96, 2, 2]
            nn.ReLU(),
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded



class Cifar_Encoder_384(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_384, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),          # [batch, 48, 4, 4]
            # nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)    # output size  [batch, 96, 2, 2]
        return encoded


class Cifar_Decoder_384(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_384, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_768_32(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_768_32, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            nn.ReLU(),
			#nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            #nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)          # Output Size: [batch, 48, 4, 4]
        return encoded


class Cifar_Decoder_768_32(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_768_32, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			#nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            #nn.ReLU(),
			nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_768_24(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_768_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            #nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)          # Output Size: [batch, 48, 4, 4]
        return encoded


class Cifar_Decoder_768_24(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_768_24, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_1536(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_1536, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            #nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Cifar_Decoder_1536(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_1536, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Cifar_Encoder_2048(nn.Module):
    def __init__(self):
        super(Cifar_Encoder_2048, self).__init__()
        # Input size: [batch, 3, 32, 32]  3072
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=4, padding=0),            # [batch, 12, 16, 16]
            nn.ReLU(),
            #nn.Conv2d(32, 48, 3, stride=3, padding=2),           # [batch, 24, 8, 8]
            #nn.ReLU(),
			# nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            # nn.ReLU(),
 			#nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
             #nn.ReLU(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Cifar_Decoder_2048(nn.Module):
    def __init__(self):
        super(Cifar_Decoder_2048, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
             #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
             #nn.ReLU(),
			# nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            # nn.ReLU(),
			#nn.ConvTranspose2d(48, 32, 3, stride=3 , padding=2),  # [batch, 12, 16, 16]
            #nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=4, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded
