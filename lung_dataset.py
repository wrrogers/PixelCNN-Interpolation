import os
import sys
import numpy as np
import cv2
import concurrent
from random import randint
from tqdm import tqdm
from PIL import Image,ImageOps,ImageEnhance

from sklearn.model_selection import train_test_split

from torch.nn import Parameter
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader,Subset

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

def get_images(img_path = r'C:\Users\W.Rogers\Desktop\Data\Lung1', max = False):
    folders = listfolders(img_path)
    #print(folders)
    if max:
        length=max
    else:
        length=len(folders)
    scan = []
    with tqdm(total=length) as pbar:
        pbar.set_description('loading images')
        for n, folder in enumerate(folders):
            if max: 
                if n >= max: break
            files = os.listdir(folder)
            files = [file for file in files if file[-3:] == 'bmp']
            files = file_sort(files)
            imgs = np.empty((len(files), 128, 128)).astype(np.float32)
            #imgs = []
            for n, file in enumerate(files):
                #print(os.path.join(folder, file))
                img = cv2.imread(os.path.join(folder, file), 0)
                #img = Image.open(os.path.join(folder, file)).convert('L')
                #print(type(img))
                #print(img.shape)
                imgs[n] = img.astype(np.float32)
                #imgs.append(img)
            scan.append(imgs)
            pbar.update(1)
    
    return scan

#img_path = r'C:\Users\w.rogers\Desktop\Data\Lung1'
#folders = listfolders(img_path)
#folder = folders[164]
#files = os.listdir(folder)
#files = [file for file in files if file[-3:] == 'bmp']
#new_files = file_sort(files)
#imgs = get_images()
#print(len(imgs), imgs[0].min(), imgs[0].max())
#plt.imshow(imgs[0][10])

class LungDataset(Dataset):
    def __init__(self, path=None, init_transform=None, transform=None, split=False, train=True):
        #print("\n\nCreate a training set:", train)
        
        if path is not None:
            self.imgs = get_images(path)
        else:
            self.imgs = get_images()

        ids = np.arange(0, len(self.imgs), 1)
            
        if split:
            X_train, X_test = train_test_split(self.imgs, test_size=.02, train_size=.98, random_state=86)
            if train:
                self.imgs = X_train
            else:
                #print('using test data')
                self.imgs = X_test            
        
        self.batch_transform = transform
        
        print("\n... LUNG1 Dataset Intialized with", len(self.imgs), "scans")
        
            
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        imgs =  self.imgs[idx]
        print('\n-----------------------------')
        print(imgs.shape, imgs.dtype, imgs.min(), imgs.max(), imgs.std())
        n_imgs = len(imgs)
        #if n_imgs > 2:
        n = randint(1, n_imgs-2)
        #else:
        #    n = randint(1, n_imgs)
        
        #info = str(idx) + ' ' + str(n) + ' from ' + str(n-1) + ' to '+ str(n+2)
        
        imgs = imgs[n-1:n+2, :, :]
        #print(imgs.shape)
        imgs = np.rollaxis(imgs, -1)
        #print(imgs.shape)
        imgs = np.rollaxis(imgs, -1)
        #print(imgs.shape)

        if self.batch_transform:
            #print("doing transform ...")
            imgs = self.batch_transform(imgs)
        
        print(imgs.size(), imgs.dtype, imgs.min(), imgs.max(), imgs.std())
        print('-----------------------------\n')
        return imgs
     
def load_images(batch_size=32, split=False, train=True):
    while True:
        for ii, data in enumerate(create_data_loader(batch_size, split=split, train=train)):
            yield data

def create_data_loader(batch_size, split=False, train=True):
    transform = transforms.Compose([#transforms.RandomCrop(img_size),
                                     #transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0, 0, 0],std=[255, 255, 255])
                                    ])
    
    train_set = LungDataset(split=split, transform=transform, train=train)
    
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size,
                              num_workers=0, pin_memory=True)
    
    #print("The length of the training set is", len(train_set))
    return train_loader

'''
gen = load_images(batch_size=8)
img_batch = next(gen)
print("batches:", img_batch.size())
image = img_batch[0]
image = image.permute(1,2,0).detach().cpu().numpy()
print("INFO:", image.shape, image.min(), image.max())
plt.figure(figsize=(10,5))
print(image.shape)
for x in range(image.shape[-1]):
    plt.subplot(1,3,x+1)
    plt.imshow(image[:, :, x])
plt.show()
'''

#training = create_data_loader(batch_size=8, split=True, train=True)

#len(training)
#training[0]

'''
transform = transforms.Compose([ transforms.ToTensor(),
                                 transforms.Normalize(mean=[0, 0, 0],std=[255, 255, 255]),
                                 ])
      
valid_dataset = LungDataset(split=True, transform=transform, train=False) 

batch = next(iter(valid_dataset))
batch.size()
'''






