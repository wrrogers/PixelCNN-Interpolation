import os
import sys
import numpy as np
import pandas as pd
import cv2
import concurrent
from random import randint
from tqdm import tqdm
from PIL import Image,ImageOps,ImageEnhance

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image

from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt


def listfolders(path, verbose = False, absolute = True):
	filenames= os.listdir (path) # get all files' and folders' names in the current directory
	if verbose: print("Total files and folders:", len(filenames))
	result = []
	for filename in filenames: # loop through all the files and folders
		if os.path.isdir(os.path.join(os.path.abspath(path), filename)): # check whether the current object is a folder or not
			result.append(filename)
	if verbose: print("Total folders:", len(result))
	if verbose: print("Total files:", len(filenames) - len(result), '\n')
	if absolute: result = [os.path.join(path, x) for x in result]
	return result

def file_sort(files):
    nums = np.array([int(file[:file.find('.')]) for file in files])
    new_range = list(range(nums.min(), nums.max()+1))
    new_files = [str(n)+'.bmp' for n in new_range]
    return new_files

def get_images(ids, img_path = r'C:\Users\W.Rogers\Desktop\Data\CCR_new\sections', max = False):
    scans = []
    with tqdm(total=len(ids)) as pbar:
        pbar.set_description('loading images')
        for n, id in enumerate(ids):
            if max: 
                if n >= max: break
            imgs = np.load(os.path.join(img_path, id+'.npy'))
            imgs = (imgs * 255).astype(np.uint8)
            #imgs = np.rollaxis(imgs, -1)
            scans.append(imgs)          
            pbar.update(1)
    return scans

def get_ids(group=1, thickness=1.5):
    arp_file = r'C:\Users\W.Rogers\Desktop\Data\CCR_new\Acquisition_and_recon_params_CCR2.xlsx'
    arp = pd.read_excel(arp_file)
    large = arp.loc[arp['PixelSpacing'] == "['0.390625', '0.390625']", 'PatientName'].to_list()
    large.sort()
    thick = arp.loc[arp['SliceThickness'] == thickness, 'PatientName'].to_list()
    ids = os.listdir(r'C:\Users\W.Rogers\Desktop\Data\CCR_new\sections')
    groups = []
    for id in ids:
        if id[-5:] == '{}.npy'.format(str(group)):
            groups.append(id[:-4])            
    thick_ids = [group for group in groups if group[:10] in thick]   
    large_ids = [group for group in groups if group[:10] in large]
    ids = [id for id in thick_ids if id in large_ids]
    return ids

'''
ids = get_ids(group=2)
print(ids)
scans = get_images(ids)
print(len(scans))
for x in range(len(scans)):
    print(len(scans), scans[x].min(), scans[x].max(), scans[x].shape)

for x in range(len(scans)):
    plt.figure(figsize=(16,3))
    scan = scans[x]
    for n in range(scan.shape[-1]):
        plt.subplot(1,scan.shape[-1],n+1)
        plt.imshow(scan[:,:,n])
    plt.show()
'''

class PhantomDataset(Dataset):
    def __init__(self, ids, dims=(64, 64), path=None, transform=None, split=False, train=True, gap=False):
        print("\n\nCreate a training set:", train)
        self.dims = dims
        self.gap = gap
        
        if path is not None:
            self.imgs = get_images(ids, path=path)
        else:
            self.imgs = get_images(ids)

        #ids = np.arange(0, len(self.imgs), 1)
        if split:
            X_train, X_test = self.imgs[:-1], self.imgs[-1:]
            if train:
                self.imgs = X_train
            else:
                #print('using test data')
                self.imgs = X_test            
        
        self.batch_transform = transform
        
        print("\n... Phantom Dataset Intialized with", len(self.imgs), "scans of shape", self.imgs[0].shape)
            
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        imgs =  self.imgs[idx]
        #print('\n-----------------------------')
        #print(imgs.shape, imgs.dtype, imgs.min(), imgs.max(), imgs.std())
        n_imgs = imgs.shape[-1]
        #if n_imgs > 2:
        n = randint(1, n_imgs-2)
        #else:
        #    n = randint(1, n_imgs)
        imgs = imgs[:, :, n-1:n+2] # Select 3 images

        if not self.gap:
            temp1 = np.array(imgs[:, :, 1])
            temp2 = imgs[:, :, 2]
            imgs[:, :, 1] = temp2
            imgs[:, :, 2] = temp1
        else:
            #print('The before shape:', imgs.shape)
            #plt.figure(figsize=(9.5, 16))
            imgs = np.delete(imgs, -1, 2)
            #print('The deleted shape:', imgs.shape)
            imgs = np.pad(imgs, (0, 1), 'constant')
            #print('Appended new shape:', imgs.shape)
            imgs = imgs[:imgs.shape[0]-1, :imgs.shape[1]-1, :]
            #print('Appended new shape:', imgs.shape)
            #imgs[:, :, -1] = imgs[:, :, 1]
            #imgs[:, :, 1] = 0
        
        #for x in range(3):
        #    plt.subplot(1,3,x+1)
        #    plt.imshow(imgs[:,:,x])
        #plt.show()        
        #print('------------------------------')
        #print(imgs.shape)
        #imgs = np.rollaxis(imgs, -1)
        #print(imgs.shape)
        #imgs = np.rollaxis(imgs, -1)
        #print(imgs.shape)
        imgs = Image.fromarray(imgs)
        #imgs = image.extract_patches_2d(imgs, self.dims, 1).squeeze()
        #print(imgs.shape)
        if self.batch_transform:
            #print("doing transform ...")
            imgs = self.batch_transform(imgs)
        
        #print(imgs.shape, imgs.size())
        #print(imgs.size(), imgs.dtype, imgs.min(), imgs.max(), imgs.std())
        #print('-----------------------------\n')

        return imgs
     
def load_images(ids, batch_size=8, split=False, train=True):
    while True:
        for ii, data in enumerate(create_data_loader(ids, batch_size, split=split, train=train)):
            yield data

def create_data_loader(ids, batch_size, img_size=(64,64), split=False, train=False):
    transform = transforms.Compose([ transforms.RandomCrop(img_size),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[.5, .5, .5],std=[.5, .5, .5])
                                    ])
    
    train_set = PhantomDataset(ids, dims=img_size, split=split, transform=transform, train=train)
    
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size,
                              num_workers=0, pin_memory=True)
    
    #print("The length of the training set is", len(train_set))
    return train_loader


#training = create_data_loader(batch_size=8, split=True, train=True)
#len(training)

ids = get_ids(group=0)

'''
groups = [0, 1, 2]
ids = np.array([])
for group in groups:
    id = get_ids(group=group)
    ids = np.append(ids, id)

scans = load_images(ids, split=False, train=False)

batch = next(scans)

data = batch.detach().cpu().numpy()
data.shape
print(data.min(), data.max())
for n in range(len(batch)):
    for x in range(3):
        plt.subplot(1,3,x+1)
        plt.imshow(data[n,x,:,:])
    plt.show()
'''





