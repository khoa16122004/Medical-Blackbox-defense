import torch
import torch.nn as nn

class MSE_CE_loss(nn.Module):
    def __init__(self, alpha=0.1):
        super(MSE_CE_loss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x_denoise, x, output, label):
        loss_mse = self.mse_loss(x_denoise, x)
        
        loss_ce = self.ce_loss(output, label)

        loss = self.alpha * loss_ce + loss_mse

        return loss

