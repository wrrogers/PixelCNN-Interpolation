"""
PixelCNN++ implementation following https://github.com/openai/pixel-cnn/

References:
    1. Salimans, PixelCNN++ 2017
    2. van den Oord, Pixel Recurrent Neural Networks 2016a
    3. van den Oord, Conditional Image Generation with PixelCNN Decoders, 2016c
    4. Reed 2016 http://www.scottreed.info/files/iclr2017.pdf
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

from tqdm import tqdm
import  matplotlib.pyplot as plt

# --------------------
# Helper functions
# --------------------

def down_shift(x):
#    B, C, H, W = x.shape
#    return torch.cat([torch.zeros([B, C, 1, W], device=x.device), x[:,:,:H-1,:]], 2)
    print()
    print('Shift down ...')
    shifted = F.pad(x, (0,0,1,0))[:,:,:-1,:]
    temp_x = x.detach().squeeze().cpu()
    for n in range(4):
        plt.subplot(1,4,n+1)
        plt.imshow(temp_x[n])
    plt.show()
    temp_shifted = shifted.detach().squeeze().cpu()
    for n in range(4):
        plt.subplot(1,4,n+1)
        plt.imshow(temp_shifted[n].detach().squeeze().cpu())
    plt.show()
    print('DONE\n')
    return shifted

def right_shift(x):
#    B, C, H, W = x.shape
#    return torch.cat([torch.zeros([B, C, H, 1], device=x.device), x[:,:,:,:W-1]], 3)
    print()
    print('Shift right ...')
    shifted = F.pad(x, (1,0))[:,:,:,:-1]
    temp_x = x.detach().squeeze().cpu()
    for n in range(4):
        plt.subplot(1,4,n+1)
        plt.imshow(temp_x[n])
    plt.show()
    temp_shifted = shifted.detach().squeeze().cpu()
    for n in range(4):
        plt.subplot(1,4,n+1)
        plt.imshow(temp_shifted[n].detach().squeeze().cpu())
    plt.show()
    print('DONE\n')
    return shifted

def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))

# --------------------
# Model components
# --------------------

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class DownShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad H above and W on each side
        Hk, Wk = self.kernel_size
        print('Kernel size ...', Hk, Wk)
        x = F.pad(x, ((Wk-1)//2, (Wk-1)//2, Hk-1, 0))
        return super().forward(x)

class DownRightShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad above and on left (ie shift input down and right)
        Hk, Wk = self.kernel_size
        print('Kernel size ...', Hk, Wk)
        x = F.pad(x, (Wk-1, 0, Hk-1, 0))
        return super().forward(x)

class DownShiftedConvTranspose2d(ConvTranspose2d):
    def forward(self, x):
        x = super().forward(x)
        _, _, Hout, Wout = x.shape
        Hk, Wk = self.kernel_size
        Hs, Ws = self.stride
#        return x[:, :, :Hout - Hk + 1, (Wk-1)//2: Wout - (Wk-1)//2]
        return x[:, :, :Hout-Hk+Hs, (Wk)//2: Wout]  # see pytorch doc for ConvTranspose output

class DownRightShiftedConvTranspose2d(ConvTranspose2d):
    def forward(self, x):
        x = super().forward(x)
        _, _, Hout, Wout = x.shape
        Hk, Wk = self.kernel_size
        Hs, Ws = self.stride
#        return x[:, :, :Hout+1-Hk, :Wout+1-Wk]  # see pytorch doc for ConvTranspose output
        return x[:, :, :Hout-Hk+Hs, :Wout-Wk+Ws]  # see pytorch doc for ConvTranspose output

class GatedResidualLayer(nn.Module):
    def __init__(self, conv, n_channels, kernel_size, drop_rate=0, shortcut_channels=None, n_cond_classes=None, relu_fn=concat_elu):
        super().__init__()
        self.relu_fn = relu_fn

        self.c1 = conv(2*n_channels, n_channels, kernel_size)
        if shortcut_channels:
            self.c1c = Conv2d(2*shortcut_channels, n_channels, kernel_size=1)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.c2 = conv(2*n_channels, 2*n_channels, kernel_size)
        if n_cond_classes:
            self.proj_h = nn.Linear(n_cond_classes, 2*n_channels)

    def forward(self, x, a=None, h=None):
        c1 = self.c1(self.relu_fn(x))
        if a is not None:  # shortcut connection if auxiliary input 'a' is given
            c1 = c1 + self.c1c(self.relu_fn(a))
        c1 = self.relu_fn(c1)
        if hasattr(self, 'dropout'):
            c1 = self.dropout(c1)
        c2 = self.c2(c1)
        if h is not None:
            c2 += self.proj_h(h)[:,:,None,None]
        a, b = c2.chunk(2,1)
        out = x + a * torch.sigmoid(b)
        return out

# --------------------
# PixelCNN
# --------------------

class PixelCNNpp(nn.Module):
    def __init__(self, image_dims=(3,28,28), n_channels=128, n_res_layers=5, n_logistic_mix=10, n_cond_classes=None, drop_rate=0.5):
        super().__init__()

        # input layers for `up` and `up and to the left` pixels
        self.u_input  = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,3))
        self.ul_input_d = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(1,3))
        self.ul_input_dr = DownRightShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,1))

        # up pass
        self.up_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)]])

        self.up_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)]])

        # down pass
        self.down_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        self.down_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        # output logistic mix params
        #   each component has 3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
        self.output_conv = Conv2d(n_channels, (3*image_dims[0]+1)*n_logistic_mix, kernel_size=1)

    def forward(self, x, h=None):
        # add channel of ones to distinguish image from padding later on
        x = F.pad(x, (0,0,0,0,0,1), value=1)

        # input layer
        u_list  = [down_shift(self.u_input(x))]
        ul_list = [down_shift(self.ul_input_d(x)) + right_shift(self.ul_input_dr(x))]

        # up pass
        for u_module, ul_module in zip(self.up_u_modules, self.up_ul_modules):
            u_list  += [u_module(u_list[-1], h=h) if isinstance(u_module, GatedResidualLayer) else u_module(u_list[-1])]
            ul_list += [ul_module(ul_list[-1], u_list[-1], h)] if isinstance(ul_module, GatedResidualLayer) else [ul_module(ul_list[-1])]

        # down pass
        u = u_list.pop()
        ul = ul_list.pop()
        for u_module, ul_module in zip(self.down_u_modules, self.down_ul_modules):
            u  = u_module(u, u_list.pop(), h) if isinstance(u_module, GatedResidualLayer) else u_module(u)
            ul = ul_module(u, torch.cat([u, ul_list.pop()],1), h) if isinstance(ul_module, GatedResidualLayer) else ul_module(ul)

        return self.output_conv(F.elu(ul))


class PixelCNNppOneLayer(nn.Module):
    def __init__(self, image_dims=(3,28,28), n_channels=8, n_cond_classes=None, drop_rate=0.5):
        super().__init__()

        # input layers for `up` and `up and to the left` pixels
        self.u_input  = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,3))
        print(self.u_input.weight.size())
        weights = self.u_input.weight.detach().cpu().numpy().squeeze()
        print(weights.shape)
        for row in range(4):
            img = weights[row]
            print(img.shape)
            plt.subplot(1,4,row+1)
            plt.imshow(weights[row, :, :])
        plt.show()
                
        self.ul_input_d = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(1,3))
        print(self.ul_input_d.weight.size())
        weights = self.ul_input_d.weight.detach().cpu().numpy()[0]
        print(weights.shape)
        for row in range(4):
            img = weights[row]
            print(img.shape)
            plt.subplot(1,4,row+1)
            plt.imshow(weights[row, :, :])
        plt.show()        
        
        self.ul_input_dr = DownRightShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,1))
        print(self.ul_input_dr.weight.size())
        weights = self.ul_input_dr.weight.detach().cpu().numpy()[0]
        print(weights.shape)
        for row in range(4):
            img = weights[row]
            print(img.shape)
            plt.subplot(1,4,row+1)
            plt.imshow(weights[row, :, :])
        plt.show()
        
        
        
    def forward(self, x, h=None):
        # add channel of ones to distinguish image from padding later on
        x  = F.pad(x, (0,0,0,0,0,1), value=1)
        print('inputing', x.size())
        u = self.u_input(x)

        return x, u
        
        
if __name__ == '__main__':
    import numpy as np
    import cv2
    
    x = cv2.imread('six.png').astype(np.float32)
    x = torch.tensor(x)
    padded = F.pad(x, (0, 1, 0, 0, 0, 0), value=0)
    padded = padded.detach().cpu().numpy()
    print(padded.shape)
    for n in range(padded.shape[-1]):
        plt.subplot(1,padded.shape[-1],n+1)
        plt.imshow(padded[:,:,n])
    plt.show()
    
    x = x.unsqueeze(0).cpu()
    x = x.permute(0,3,1,2)
    #print(x.size())
    print('Make model ...')
    model = PixelCNNppOneLayer((3,32,32), 1, 1).to('cpu')
    print()
    print('Foward ...')
    x, ds = model(x)
    x = x.squeeze().numpy()
    print(x.shape)
    #print(np.unique(x))
    for n in range(4):
        plt.subplot(1,4,n+1)
        plt.imshow(x[n])
    plt.show()
    
    '''
    u = ds.detach().squeeze().numpy()
    print(u.shape)
    for n in range(4):
        plt.subplot(1,4,n+1)
        plt.imshow(u[n])
    plt.show()
    '''
    '''
    ul = ul.detach().squeeze().numpy()
    print(ul.shape)
    for n in range(4):
        plt.subplot(1,4,n+1)
        plt.imshow(ul[n])
    plt.show()        
    ''' 
        
        
        
        
        
        
        
        
        
