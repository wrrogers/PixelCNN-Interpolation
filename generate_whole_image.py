import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

import nvgpu

import torch
from torchvision import transforms
from torch.optim import Adam

from PIL import Image

from get_windows import *
from get_windows import get_windows, get_side, get_bottom

from parameters import Parameters
from pixelcnnpp import sample_from_discretized_mix_logistic
from pixelcnnpp import PixelCNNpp, generate_fn

from phantom_control_dataset import create_data_loader, get_ids

args = Parameters()

def normalize(x):
    x = (x + x.min() + 1.)
    x = ((x-x.min())/(x.max()-x.min()))
    x = x * 2
    x = x - 1
    return x

def normalize256(x):
    x = (x + x.min() + 1.)
    x = ((x-x.min())/(x.max()-x.min()))
    x = x * 255
    x = x.astype(np.uint8)
    return x

def get_patch(img, window=(0, 0), patch_size=(64, 64)):
    half_patch = (patch_size[0]//2, patch_size[1]//2)
    #nx_patches = (img.shape[0]//patch_size[0]) + ((img.shape[0]-half_patch[0])//patch_size[0])
    #ny_patches = (img.shape[1]//patch_size[1]) + ((img.shape[1]-half_patch[1])//patch_size[1])
    
    patch = img[half_patch[0]*window[0]:half_patch[0]*window[0]+64, 
                half_patch[1]*window[1]:half_patch[1]*window[1]+64]
    patch = patch.astype(np.float32)
    patch = torch.from_numpy(patch)
    return patch

def get_corners(img, n=0, patch_size=(64, 64)):
    bottom_patch = img[-patch_size[0]:, :patch_size[1]]
    side_patch   = img[:patch_size[1], -patch_size[0]:]
    return bottom_patch, side_patch

def update_whole(img, patch, window=(0, 0), patch_size=(64, 64)):
    half_patch = (patch_size[0]//2, patch_size[1]//2)
    img[half_patch[0]*window[0]:half_patch[0]*window[0]+64, 
        half_patch[1]*window[1]:half_patch[1]*window[1]+64] = patch.detach().cpu().numpy()
    return img
    
'''
patch = get_patch(whole_img, window=(1, 0))
plt.imshow(patch)

patch = get_patch(whole_img, window=(0, 1))
plt.imshow(patch)
'''

def generate(img, xmin=0, ymin=0, zero=True):
    out = torch.tensor(img)
    print('\n\nOutput size:', out.size(), '\n')
    #print("The generate size is:", out.size())
    #targets = torch.tensor(out[:, 2, :, :].detach()).cpu().numpy().squeeze()
    if zero: out[:, 2, :, :] = torch.zeros(64,64)
    
    '''
    for img in range(img.size(0)):
        for x in range(3):
            plt.subplot(1,3,x+1)
            print('out', out.size())
            temp = np.moveaxis(out[img, x, :, :].detach().cpu().numpy(), 0, -1)
            
            plt.imshow(temp)
        plt.show()
    '''
    
    with tqdm(total=(image_dims[1]-xmin)*(image_dims[2]-ymin), desc='Generating {} image(s)'.format(out.size(0))) as pbar:
        for yi in range(image_dims[1]):
            if yi < ymin: 
                #print(yi)
                continue
            for xi in range(image_dims[2]):
                if xi < xmin: 
                    #print(xi)
                    continue
                logits = model(out, h)
                sample = sample_from_discretized_mix_logistic(logits, image_dims)[:,:,yi,xi]
                print('Sample type:', sample.dtype, sample.size(), sample)
                pixel = torch.tensor(sample[:,2])
                print('Pixel type', pixel.dtype, pixel.size(), pixel)
                out[:,2,yi,xi] = pixel

                for base in range(out.size(0)):
                    plt.figure(figsize=(16, 16))
                    for i in range(3):
                        plt.subplot(1, 3, i+1)
                        temp = np.moveaxis(out[base, i].detach().cpu().numpy().squeeze(), 0, -1).T
                        plt.imshow(temp)
                    plt.show()
                
                del sample, logits
                gc.collect()
                pbar.update()
    return out#, targets

n = 8
image_dims = (3, 64, 64)
h = None

id = get_ids(0)[-1]
model_path = r'C:\Users\william\PixelCNN\PixelCNN-Interpolation\results\pixelcnnpp\2020-05-12_06-44-27'
model_file = 'checkpoint.pt'
path = r'H:\Data\CCR_new\numpy'
file = id+'.npy'
scan = np.rollaxis(np.load(os.path.join(path, file)), -1)

img_size = scan.shape[1:]

print('All scan shape:    ', scan.shape)

print('The shape from the scan', scan.shape[0])

# Grab the selected slice
imgs = scan[n-1:n+2, :, :]

# Move the middle slice to the last position
temp1 = np.array(imgs[1, :, :])
temp2 = imgs[2, :, :]
imgs[1, :, :] = temp2
imgs[2, :, :] = temp1

print('Scan layer', n, 'shape:', imgs.shape)
windows = get_windows(imgs)
windows = np.moveaxis(windows, 2, -1)

print('Shape of the windows:', windows.shape)

'''
fig, axs = plt.subplots(windows.shape[0], windows.shape[1], sharex='col', sharey='row',
                        gridspec_kw={'hspace':.02, 'wspace':.02}, figsize=(16, 16))
count = 0
for n in range(windows.shape[0]):
    for m in range(windows.shape[1]):
        axs[n, m].imshow(np.moveaxis(windows[n, m, :, :], 0, -1))
        count+=1
plt.show()
'''

model = PixelCNNpp(args.image_dims, args.n_channels, args.n_res_layers, 
                   args.n_logistic_mix, args.n_cond_classes).to(args.device)

model = torch.nn.DataParallel(model).to(args.device)

checkpoint = torch.load(os.path.join(model_path, model_file))
#checkpoint['state_dict'].keys()
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.cuda()
model.eval()


''' Do the first window '''
x = windows[0, 0, :, :, :]

plt.figure(figsize=(10,10))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(x[:, :, i])
plt.show()

print(x.min(), x.max())
x = normalize256(x)
print(x.min(), x.max())

x = Image.fromarray(x)
transform = transforms.Compose([ transforms.ToTensor(),
                                 transforms.Normalize(mean=[.5, .5, .5],std=[.5, .5, .5]) ])

x = transform(x)

x.size()
print(x.min(), x.max())
out = generate(x)

out.size()
out_img, targets = out.detach().cpu().numpy().squeeze()
plt.figure(figsize=(8, 8))
plt.imshow(out_img[2])
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(x[-1, :, :])
plt.show()

neg = out_img[2] - x[-1, :, :].detach().cpu().numpy().squeeze()
plt.figure(figsize=(8, 8))
plt.imshow(neg)
plt.show()

#np.save('First.npy', out)

''' Do the first row and column windows '''

out = np.load('First.npy', allow_pickle = True)
out_img, targets = out
out_img =out_img.detach().cpu().numpy().squeeze()
plt.figure(figsize=(8, 8))
plt.imshow(out_img[2])
plt.show()

#whole_img = np.empty(img_size, dtype=np.float32)
whole_img = np.kron([[1, 0] * 3, [0, 1] * 3] * 3, np.ones((64, 64)))[:204, :204]
plt.imshow(whole_img)
whole_img[:64, :64] = out_img[2]
plt.imshow(whole_img)

for n in range(windows.shape[0]):
    if n > 0:
        x = windows[0, n, :, :, :]
        x = normalize256(x)
        x = transform(x)
        x[-1, :, :] = get_patch(whole_img, (0,n))
        plt.imshow(x[0].detach().cpu().numpy().squeeze())
        
        out, target = generate(x, xmin=32, zero=False)
        
        whole_img = update_whole(whole_img, out.squeeze()[-1, :, :], window=(0,n))
        
        x = windows[n, 0, :, :, :]
        x = normalize256(x)
        x = transform(x)
        x[-1, :, :] = get_patch(whole_img, (n,0))
        plt.imshow(x[0].detach().cpu().numpy().squeeze())
        
        out, target = generate(x, ymin=32, zero=False)
        
        whole_img = update_whole(whole_img, out.squeeze()[-1, :, :], window=(n,0))
        
        plt.imshow(whole_img)
        plt.show()
        
        #np.save('sides.npy', whole_img)

''' Do the internal windows '''
whole_img = np.load('sides.npy')

for n in range(1, windows.shape[0]):
    print(n, n)
    x = windows[n, n, :, :, :]
    x = normalize256(x)
    x = transform(x)
    x = x.unsqueeze(0).cuda()
    x[:, -1, :, :] = get_patch(whole_img, (n,n))
    out = generate(x, xmin=32, ymin=32, zero=False)
    #print(out.shape)
    whole_img = update_whole(whole_img, out.squeeze()[-1, :, :], window=(n,n))
    plt.imshow(whole_img)
    plt.show()
    for m in range(n, windows.shape[0]):
        if m > n:
            print(n, m)
            print(m, n)
            
            horizontal = windows[n, m, :, :, :]
            horizontal = normalize256(horizontal)
            horizontal = transform(horizontal)
            
            vertical = windows[m, n, :, :, :]
            vertical = normalize256(vertical)
            vertical = transform(vertical)
            
            #print(horizontal.size(), vertical.size())
            
            x = torch.stack([horizontal, vertical], axis=0)
    
            #print(x.size())
    
            x[0, -1, :, :] = get_patch(whole_img, (n,m))
            x[1, -1, :, :] = get_patch(whole_img, (m,n))
            
            #print(x.size())
    
            out = generate(x, xmin=32, ymin=32, zero=False)
            
            whole_img = update_whole(whole_img, out[0, -1, :, :], window=(n,m))
            whole_img = update_whole(whole_img, out[1, -1, :, :], window=(m,n))
            
            plt.imshow(whole_img)
            plt.show()
    
#np.save('sides2.npy', whole_img)
    
''' Do the side windows windows '''

side_windows   = get_side(imgs)
side_whole_img = get_side(np.expand_dims(whole_img, 0), dtype=np.float64)
side_windows[:, -1, :, :] = side_whole_img.squeeze()

for n in range(side_windows.shape[0]):
    side = np.moveaxis(side_windows[n], 0, -1)
    side = normalize256(side)
    side = transform(side)
    print('min max:', side.min(), side.max(), side.dtype)
    side = side.unsqueeze(0)
    #xmin = 64 - (imgs.shape[1] - (32*side_windows.shape[0]+32))

    out = generate(side, zero=True)

''' Do the side windows windows '''

bottom_windows = get_bottom(imgs)
bottom_whole_img = get_bottom(np.expand_dims(whole_img, 0), dtype=np.float64)
bottom_windows[:, -1, :, :] = bottom_whole_img.squeeze()

for n in range(side_windows.shape[0]):
    bottom = np.moveaxis(bottom_windows[n], 0, -1)
    bottom = normalize256(bottom)
    bottom = transform(bottom)



''' Do the corner window '''


    

        









