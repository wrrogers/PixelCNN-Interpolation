import os
import sys
import numpy as np
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

def get_images(img_path = r'C:\Users\W.Rogers\Desktop\Data\CCR_new\sections', max = False):
    files = os.listdir(img_path)
    #print(files)
    files = [file for file in files if file[-3:] == 'npy']
    if max:
        length=max
    else:
        length = len(files)
    #print(files)
    #files = file_sort(files)
    scans = []
    with tqdm(total=length) as pbar:
        pbar.set_description('loading images')
        for n, file in enumerate(files):
            if max: 
                if n >= max: break
            imgs = np.load(os.path.join(img_path, file))
            imgs = (imgs * 255).astype(np.uint8)
            #imgs = np.rollaxis(imgs, -1)
            scans.append(imgs)          
            pbar.update(1)
    return scans

#scans = get_images(max = 10)
#print(len(scans))
#print(len(scans), scans[0].min(), scans[0].max(), scans[0].shape)
#plt.imshow(scans[0][:, :, 0])
#imgs = scans[0][:, :, :3]

class PhantomDataset(Dataset):
    def __init__(self, dims=(64, 64), path=None, transform=None, split=False, train=True):
        print("\n\nCreate a training set:", train)
        self.dims = dims
        
        if path is not None:
            self.imgs = get_images(path)
        else:
            self.imgs = get_images()

        #ids = np.arange(0, len(self.imgs), 1)
            
        if split:
            X_train, X_test = train_test_split(self.imgs, test_size=.01, train_size=.99, random_state=86)
            if train:
                self.imgs = X_train
            else:
                #print('using test data')
                self.imgs = X_test            
        
        self.batch_transform = transform
        
        print("\n... Phantom Dataset Intialized with", len(self.imgs), "scans")
            
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
        
        #info = str(idx) + ' ' + str(n) + ' from ' + str(n-1) + ' to '+ str(n+2)
        
        imgs = imgs[:, :, n-1:n+2]
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
     
def load_images(batch_size=8, split=False, train=True):
    while True:
        for ii, data in enumerate(create_data_loader(batch_size, split=split, train=train)):
            yield data

def create_data_loader(batch_size, img_size=(64,64), split=False, train=False):
    transform = transforms.Compose([ transforms.RandomCrop(img_size),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     #transforms.Normalize(mean=[0, 0, 0],std=[1, 1, 1])
                                    ])
    
    train_set = PhantomDataset(dims=img_size, split=split, transform=transform, train=train)
    
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size,
                              num_workers=0, pin_memory=True)
    
    #print("The length of the training set is", len(train_set))
    return train_loader


#training = create_data_loader(batch_size=8, split=True, train=True)
#len(training)

'''
scans = load_images()

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