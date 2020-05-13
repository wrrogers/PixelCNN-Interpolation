import torch
from torch import nn, optim
from torch import autograd
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader,Subset

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        #print("Q Input:", input.size())
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)
        #print("Embeded ind:", embed_ind.size())
        #print("Embed Code: ", quantize.size())
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock_D(nn.Module):
    def __init__(self, in_channel, channel, params):
        super().__init__()
        if params['spectral_resblock']:
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(in_channel, channel, 3, padding=1)),
                nn.ReLU(inplace=True),
                nn.utils.spectral_norm(nn.Conv2d(channel, in_channel, 1)),
            )
        else:
            self.conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, channel, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, in_channel, 1),
            )
            
    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class ResBlock_E(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )
            
    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        #print("Intitalizing Encoder ...")
        #print("In channel:", in_channel, ", Channel:", channel)
        #print("Number of res blocks:  ", n_res_block)
        #print("Number of res channels:", n_res_channel)
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock_E(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)
        #print()

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, params
    ):
        super().__init__()
        #print("Initializing Decoder ...")
        #print("In channel:", in_channel, ", Out channel:", out_channel, ", Channel:", channel)
        #print("Number of res blocks:  ", n_res_block)
        #print("Number of res channels:", n_res_channel)
        
        if params['spectral_first_layer']:
            blocks = [nn.utils.spectral_norm(nn.Conv2d(in_channel, channel, 3, padding=1))]
        else:
            blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        if params['attention_1']:
            blocks.append(Attention(params['attention_1_size']))

        for i in range(n_res_block):
            blocks.append(ResBlock_D(channel, n_res_channel, params))

        blocks.append(nn.ReLU(inplace=True))
        
        if params['attention_2']:
            blocks.append(Attention(params['attention_2_size']))

        if stride == 4:
            if params['spectral_last_layer']:
                blocks.extend(
                    [
                        nn.utils.spectral_norm(nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1)),
                        nn.ReLU(inplace=True),
                        nn.utils.spectral_norm(nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1)),
                    ]
                )
            else:
                blocks.extend(
                    [
                        nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                    ]
                )

        elif stride == 2:
            if params['spectral_last_layer']:
                blocks.append(
                    nn.utils.spectral_norm(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
                )
            else:
                blocks.append(
                    nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
                )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        params,
        in_channel=2,
        out_channel=1,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
                embed_dim, 
                embed_dim, 
                channel, 
                n_res_block, 
                n_res_channel, 
                stride=2,
                params=params
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            out_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
            params=params
        )
        #print("Big VQVAE initialized.")

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        #print("Forward")
        #print("-------------------------------------------")
        #print("Quantization T:", quant_t.size())
        #print("Quantization B:", quant_b.size())
        #print("Difference:", diff)
        #print("-------------------------------------------")
        dec = self.decode(quant_t, quant_b)
        return dec, diff

    def encode(self, input):
        #print("-------------------------------------------")
        #print("Encode")
        #print("-------------------------------------------")
        enc_b = self.enc_b(input)
        #print("Encoding B:", enc_b.size())
        enc_t = self.enc_t(enc_b)
        #print("Encoding T:", enc_b.size())

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        #print("Encoding T after conv:", quant_t.size())
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        #print("Encoding T quantized:", quant_t.size())
        quant_t = quant_t.permute(0, 3, 1, 2)
        #print("Quantized T rolled:", quant_t.size())
        
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        #print("Decoding T:", dec_t.size())
        enc_b = torch.cat([dec_t, enc_b], 1)
        #print("Cat Dec T and Enc B:", enc_b.size())

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)
        
        #print()
        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        #print()
        #print("Decode")
        #print("-------------------------------------------")
        #print("Upsample", upsample_t.size())
        quant = torch.cat([upsample_t, quant_b], 1)
        #print("Upsample", quant.size())
        dec = self.dec(quant)
        #print("Decoded:", dec.size())
        #print()
        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec

class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta    = nn.utils.spectral_norm(self.conv1x1(channels, channels//8))
        self.phi      = nn.utils.spectral_norm(self.conv1x1(channels, channels//8))
        self.g        = nn.utils.spectral_norm(self.conv1x1(channels, channels//2))
        self.o        = nn.utils.spectral_norm(self.conv1x1(channels//2, channels))
        self.gamma    = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def forward(self, inputs):
        batch,c,h,w = inputs.size()
        #print("Attention input", inputs.size())
        theta = self.theta(inputs) #->(*,c/8,h,w)
        phi   = F.max_pool2d(self.phi(inputs), [2,2]) #->(*,c/8,h/2,w/2)
        g     = F.max_pool2d(self.g(inputs), [2,2]) #->(*,c/2,h/2,w/2)
        
        theta = theta.view(batch, self.channels//8, -1) #->(*,c/8,h*w)
        phi   = phi.view(batch, self.channels//8, -1) #->(*,c/8,h*w/4)
        g     = g.view(batch, self.channels//2, -1) #->(*,c/2,h*w/4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1) #->(*,h*w,h*w/4)
        o    = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch,self.channels//2,h,w)) #->(*,c,h,w)
        return self.gamma*o + inputs
    
    def conv1x1(self, in_channel, out_channel): #not change resolution
        return nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0,dilation=1,bias=False)
















