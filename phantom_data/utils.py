import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nrrd


def crop(arr):
    #print("The shape of the array was:", arr.shape[:2])
    #Chop z axis with all zeros
    xd = []
    for x in range(arr.shape[0]):
        if arr[x,:,:].sum() == 0:
            xd.append(x)
    arr = np.delete(arr,xd,axis=0)
    #Chop y axis with all zeros
    yd = []
    for y in range(arr.shape[1]):
        if arr[:,y,:].sum() == 0:
            yd.append(y)
    arr = np.delete(arr,yd,axis=1)
    #print("The shape of the new array is:", arr.shape[:2])
    return arr


def get_paths(group, src_dir):
    paths = []
    subsets  = os.listdir(src_dir)
    subset = subsets[group]
    #print('Base paths:', src_dir)
    #print('Subset:    ', subset)

    scans = os.listdir(os.path.join(src_dir, subset))
    for scan in scans:
        paths.append(os.path.join(src_dir, subset, scan))
    return paths, subset


def get_images(group,
               scan_number, 
               src_dir = r'C:\Users\W.Rogers\Desktop\Data\CCR_new',):
    
    paths, subset = get_paths(group, src_dir)
    path  = paths[scan_number]
    group = path[-19:11]
    id    = path[-10:]
    #print('ID:        ', id)
    scan, _ = nrrd.read(path+'\image.nrrd')
    #print("Scan dims: ", scan.shape)

    scan    = np.abs(scan)
    scan    = (scan-scan.min())/(scan.max() - scan.min())
    
    mask, _ = nrrd.read(path+'\mask1.nrrd')
    mask[mask>0] = 1
    scan = scan*mask
    scan = crop(scan)

    d = []
    for i in range(mask.shape[-1]):
        drop = len(np.unique(mask[:,:,i])) == 1
        if drop:
            d.append(i)
    scan = np.delete(scan, d, -1)
    #print('Reduced:   ', scan.shape)
    return scan, id


def show_all(scans, columns, figsize=(16, 16)):
    rows = scans.shape[-1] // columns
    if scans.shape[-1] % columns > 0: rows+=1
    plt.figure(figsize=figsize)
    for n in range(scans.shape[-1]):
        plt.subplot(rows, columns, n+1)
        plt.imshow(scans[:,:,n])
    plt.show()


def run_all(group=4, src_dir = r'C:\Users\W.Rogers\Desktop\Data\CCR_new'):
    for n in range(40):
        scans, id = get_images(group, n)
        
    
def save_all_images(group,
                    columns = 10,
                    figsize = (16, 16),
                    src_dir = r'C:\Users\W.Rogers\Desktop\Data\CCR_new',
                    sav_dir = r'C:\Users\W.Rogers\Desktop\Data\CCR_new\imgs'):
    for n in range(40):
        scans, id = get_images(group, n)
        rows = scans.shape[-1] // columns
        if scans.shape[-1] % columns > 0: rows+=1
        plt.figure(figsize=figsize)
        plt.suptitle(str(id)+' - '+str(scans.shape))
        for n in range(scans.shape[-1]):
            plt.subplot(rows, columns, n+1)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.imshow(scans[:,:,n])      

        plt.axis('off')
        plt.savefig(sav_dir+'/'+id+'.jpg', bbox_inches='tight')
        
        
def save_images(images, base_dir, id, n, x):
    plt.figure(figsize=(16, 6))
    plt.suptitle(str(id)+' - '+str(images.shape))
    for i in range(images.shape[-1]):
        plt.subplot(1, images.shape[-1], i+1)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.imshow(images[:,:,i])
    plt.axis('off')
    plt.savefig(base_dir+'/section images/'+id+'_'+str(n)+'_'+str(x)+'.jpg', bbox_inches='tight')


def strip_images(base_path = r'C:\Users\W.Rogers\Desktop\Data\CCR_new'):
    
    sect_path = os.path.join(base_path, 'sections.tsv')
    issu_path = os.path.join(base_path, 'issues.tsv')
    
    sections  = pd.read_csv(sect_path, delimiter='\t')
    sections['Sections'] = sections.Sections.apply(lambda x: x[1:-1].split(','))
    sections['Sections'] = sections.Sections.apply(lambda x: [int(item.strip()) for item in x])
    
    issues = pd.read_csv(issu_path, delimiter='\t')
    issues['Issue Images'] = issues['Issue Images'].apply(lambda x: x[1:-1].split(','))
    issues['Issue Images'] = issues['Issue Images'].apply(lambda x: [item.strip() for item in x])
    issues['Issue Images'] = issues['Issue Images'].apply(lambda x: [int(item) for item in x])
    
    for n in range(6):
        #if n > 0: break
        print('Subset:', n)
        for x in range(40):
            #if x > 2: break
            print('Group:', x)
            images, id = get_images(n, x)
            sec = np.asarray(sections.loc[sections['Scan Identity'] == id, 'Sections'].values[0])
            iss = np.asarray(issues.loc[issues['Scan Identity'] == id, 'Issue Images'].values[0]) - 1
            
            #stack = []
            
            b = 0
            t = sec[0]
            row = np.empty((images[0].shape[:2]+(t-b,)))
            row = images[:,:,b:t]
            if bool(len({*iss} & {*range(b, t)})): # Check to see if there is a bad image in the group
                i = np.asarray([x for x in iss if x in range(b, t)])
                print(range(b, t), i, i-b)
                print(row.shape)
                row = np.delete(row, i-b, 2)
                print(row.shape)
            file_name=id+'_'+str(n)+'_'+str(0)+'.npy'
            np.save(os.path.join(base_path, 'sections', file_name), row)
            save_images(row, base_path, id, x, 0)
            #stack.append(row)
            
            for s in range(0, 9):
                b=t
                t+=sec[s+1]
                row = np.empty((images[0].shape[:2]+(t-b,)))
                row = images[:,:,b:t]
                if bool(len({*iss} & {*range(b, t)})): # Check to see if there is a bad image in the group
                    i = np.asarray([x for x in iss if x in range(b, t)])
                    print(range(b, t), i, i-b)
                    row = np.delete(row, i[0]-b, 2)
                file_name=id+'_'+str(n)+'_'+str(s+1)+'.npy'
                np.save(os.path.join(base_path, 'sections', file_name), row)
                save_images(row, base_path, id, x, s+1)
                #stack.append(row)

strip_images()













