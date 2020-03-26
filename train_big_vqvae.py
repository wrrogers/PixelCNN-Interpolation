import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

#from big_vqvae import VQVAE

import matplotlib.pyplot as plt
import numpy as np

def train_vqvae_model(epoch, loader, model, optimizer, device):
    
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, data in enumerate(loader):
        #if i > 0: break
    
        model.zero_grad()

        x = data.to(device)
        #print(x.size())
        
        X  = x[:, 0::2, :, :].float()
        X̂  = x[:,    1, :, :].unsqueeze(1).float()

        out, latent_loss = model(X)
        out_imgs = out.detach().cpu().numpy()
        #print("OUT :", out_imgs.shape)
        #print("IMGS:", imgs.size())
        #img = np.moveaxis(out_imgs, 1, -1)[0]
        #plt.imshow(img)
        #print(latent_loss)
        recon_loss = criterion(out, X̂)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()
        
        optimizer.step()

        mse_sum += recon_loss.item() * X̂.shape[0]
        mse_n += X̂.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if epoch % 32 == 0:
            model.eval()

            sample = X[:sample_size]
            target = X̂[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            #print(target.size(), out.size())
            utils.save_image(
                torch.cat([target, out], 0),
                f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            model.train()
            
    return loss.detach().cpu().numpy()