import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST, CIFAR10
from torchvision import datasets, transforms

import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

import os
import pickle
import time
import datetime
import json
import pprint
from functools import partial

from torch.optim import Adam, RMSprop

from parameters import Parameters

import matplotlib.pyplot as plt

# --------------------
# Data
# --------------------

def fetch_dataloaders(args):
    # preprocessing transforms
    transform = T.Compose([T.ToTensor(),                                            # tensor in [0,1]
                           lambda x: x.mul(255).div(2**(8-args.n_bits)).floor(),    # lower bits
                           partial(preprocess, n_bits=args.n_bits)])                # to model space [-1,1]
    target_transform = (lambda y: torch.eye(args.n_cond_classes)[y]) if args.n_cond_classes else None

    if args.dataset=='mnist':
        args.image_dims = (1,28,28)
        train_dataset = MNIST(args.data_path, train=True, transform=transform, target_transform=target_transform)
        valid_dataset = MNIST(args.data_path, train=False, transform=transform, target_transform=target_transform)
    elif args.dataset=='cifar10':
        args.image_dims = (3,32,32)
        train_dataset = CIFAR10(args.data_path, train=True, transform=transform, target_transform=target_transform, download=True)
        valid_dataset = CIFAR10(args.data_path, train=False, transform=transform, target_transform=target_transform, download=True)
    elif args.dataset=='colored-mnist':
        args.image_dims = (3,28,28)
        # NOTE -- data is quantized to 2 bits and in (N,H,W,C) format
        with open(args.data_path, 'rb') as f:  # return dict {'train': np array; 'test': np array}
            data = pickle.load(f)
        # quantize to n_bits to match the transforms for other datasets and construct tensors in shape N,C,H,W
        train_data = torch.from_numpy(np.floor(data['train'].astype(np.float32) / (2**(2 - args.n_bits)))).permute(0,3,1,2)
        valid_data = torch.from_numpy(np.floor(data['test'].astype(np.float32) / (2**(2 - args.n_bits)))).permute(0,3,1,2)
        # preprocess to [-1,1] and setup datasets -- NOTE using 0s for labels to have a symmetric dataloader
        train_dataset = TensorDataset(preprocess(train_data, args.n_bits), torch.zeros(train_data.shape[0]))
        valid_dataset = TensorDataset(preprocess(valid_data, args.n_bits), torch.zeros(valid_data.shape[0]))
    elif args.dataset=='lung':
        from lung_dataset import LungDataset
        args.image_dims = (3,128,128)
        transform = transforms.Compose([ T.ToTensor(),
                                         T.Normalize(mean=[0, 0, 0],std=[255, 255, 255]),
                                         lambda x: x.mul(255).div(2**(8-args.n_bits)).floor(),
                                         partial(preprocess, n_bits=args.n_bits)
                                         ])
        train_dataset = LungDataset(split=True, transform=transform, train=True)        
        valid_dataset = LungDataset(split=True, transform=transform, train=False)
    elif args.dataset=='phantom':
        from phantom_dataset import PhantomDataset
        args.image_dims = (3,64,64)
        transform = transforms.Compose([ transforms.RandomCrop((64,64)),
                                         #transforms.RandomHorizontalFlip(p=0.5),
                                         T.ToTensor(),
                                         #T.Normalize(mean=[0, 0, 0],std=[255, 255, 255]),
                                         lambda x: x.mul(255).div(2**(8-args.n_bits)).floor(),
                                         partial(preprocess, n_bits=args.n_bits)
                                         ])
        train_dataset = PhantomDataset(split=True, transform=transform, train=True)        
        valid_dataset = PhantomDataset(split=True, transform=transform, train=False)
    elif args.dataset=='phantom_control_seen':
        from phantom_control_dataset import PhantomDataset, get_ids
        args.image_dims = (3,64,64)
        '''
        groups = [0, 1]
        ids = np.array([])
        for group in groups:
            id = get_ids(group=group)
            ids = np.append(ids, id)
        '''
        ids = get_ids(group=0)
        transform = transforms.Compose([ transforms.RandomCrop((64, 64)),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[.5, .5, .5],std=[.5, .5, .5])
                                        ])
        train_dataset = PhantomDataset(ids=ids, transform=transform, split=True, train=True, gap=False)
        valid_dataset = PhantomDataset(ids=ids, transform=transform, split=True, train=False, gap=False)
    elif args.dataset=='phantom_control_unseen':
        from phantom_control_dataset import PhantomDataset, get_ids
        args.image_dims = (3,64,64)
        ids = get_ids(group=0, thickness=1.5)
        transform = transforms.Compose([ transforms.RandomCrop((64, 64)),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[.5, .5, .5],std=[.5, .5, .5])
                                        ])
        train_dataset = PhantomDataset(ids=ids, transform=transform, split=False)
        ids = get_ids(group=0, thickness=3)
        valid_dataset = PhantomDataset(ids=ids, transform=transform, split=False, gap=True)
    else:
        raise RuntimeError('Dataset not recognized')

    if args.mini_data:  # dataset to a single batch
        if args.dataset=='colored-mnist':
            train_dataset = train_dataset.tensors[0][:args.batch_size]
        else:
            train_dataset.data = train_dataset.data[:args.batch_size]
            train_dataset.targets = train_dataset.targets[:args.batch_size]
        valid_dataset = train_dataset

    print('Dataset {}\n\ttrain len: {}\n\tvalid len: {}\n\tshape: {}\n\troot: {}'.format(
        args.dataset, len(train_dataset), len(valid_dataset), train_dataset[0][0].shape, args.data_path))

    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=(args.device.type=='cuda'), num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=(args.device.type=='cuda'), num_workers=0)

    # save a sample
    data_sample = next(iter(train_dataloader))[0]
    writer.add_image('data_sample', make_grid(data_sample, normalize=True, scale_each=True), args.step)
    save_image(data_sample, os.path.join(args.output_dir, 'data_sample.png'), normalize=True, scale_each=True)

    return train_dataloader, valid_dataloader

def preprocess(x, n_bits):
    # 1. convert data to float
    # 2. normalize to [0,1] given quantization
    # 3. shift to [-1,1]
    return x.float().div(2**n_bits - 1).mul(2).add(-1)

def deprocess(x, n_bits):
    # 1. shift to [0,1]
    # 2. quantize to n_bits
    # 3. convert data to long
    return x.add(1).div(2).mul(2**n_bits - 1).long()

def save_json(data, filename, args):
    with open(os.path.join(args.output_dir, filename + '.json'), 'w') as f:
        json.dump(data, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# --------------------
# Train, evaluate, generate
# --------------------

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, epoch, writer, args):
    model.train()

    with tqdm(total=len(dataloader), desc='epoch {}/{}'.format(epoch, args.start_epoch + args.n_epochs)) as pbar:
        #for x, info in dataloader:
        for x in dataloader:
            y = None
            args.step += 1
            
            x_in = np.load(os.path.join(r'C:\Users\W.Rogers\Desktop\Data\CCR_new\sections', 'CCR-2-0022_0_6.npy'))
            x = x_in[:64, :64, :3].astype(np.float32)
            x = np.rollaxis(x, -1)
            x = np.expand_dims(x, axis=0)
            x = (x + x.min() + 1.).astype('float32')
            x = ((x-x.min())/(x.max()-x.min()))
            x = x * 2
            x = x - 1
            x = x.astype('float32')

            #result = -1 + 2.*(data - min(data))./(max(data) - min(data));
            #print(x.shape, x.dtype)
            x = torch.tensor(x) 
            
            x = x.to(args.device)
            
            #xp = x.detach().cpu().numpy()
            #print("Mins and maxes:", xp.min(), xp.max())

            #for x in range(3):
            #    plt.subplot(1,3,x+1)
            #    plt.imshow(xp[x])
            #plt.show()
            
            logits = model(x, y.to(args.device) if args.n_cond_classes else None)
            loss = loss_fn(logits, x, args.n_bits).mean(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()

            pbar.set_postfix(loss='loss:{:.4f}, bits:{:.4f}'.format(loss.item(), loss.item() / (np.log(2) * np.prod(args.image_dims))))
            pbar.update()

            # record
            if args.step % args.log_interval == 0:
                writer.add_scalar('train_bits_per_dim', loss.item() / (np.log(2) * np.prod(args.image_dims)), args.step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], args.step)

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, args):
    model.eval()

    losses = 0
    #for x, info in tqdm(dataloader, desc='Evaluate'):
    for x in tqdm(dataloader, desc='Evaluate'):
        x = x.to(args.device)
        y = None
        logits = model(x, y.to(args.device) if args.n_cond_classes else None)
        losses += loss_fn(logits, x, args.n_bits).mean(0).item()
    return losses / len(dataloader)

@torch.no_grad()
def generate(model, data_loader, generate_fn, args):
    model.eval()
    if args.n_cond_classes:
        samples = []
        for h in range(args.n_cond_classes):
            h = torch.eye(args.n_cond_classes)[h,None].to(args.device)
            samples += [generate_fn(model, data_loader, args.n_samples, args.image_dims, args.device, h=h)]
        samples = torch.cat(samples)
    else:
        samples, targets = generate_fn(model, data_loader, args.n_samples, args.image_dims, args.device)
        #print("SAMPLE INFO:", info)
    filename = r'C:\Users\W.Rogers\PixelCNN\ii\sample\{}_S.pt'.format(str(datetime.datetime.now())[:19].replace(' ', '_').replace(':', '-'))
    torch.save(samples, filename)
    filename = r'C:\Users\W.Rogers\PixelCNN\ii\sample\{}_T.pt'.format(str(datetime.datetime.now())[:19].replace(' ', '_').replace(':', '-'))
    torch.save(targets, filename)
    samples = make_grid(samples.cpu(), normalize=True, scale_each=True, nrow=args.n_samples)
    targets = make_grid(targets, normalize=True, scale_each=True, nrow=args.n_samples)
    targets = targets.unsqueeze(1)
    return samples, targets

def train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn, writer, args):
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        # train
        train_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, epoch, writer, args)

        if (epoch+1) % args.eval_interval == 0:
            # save model
            torch.save({'epoch': epoch,
                        'global_step': args.step,
                        'state_dict': model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint.pt'))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, 'optim_checkpoint.pt'))
            if scheduler: torch.save(scheduler.state_dict(), os.path.join(args.output_dir, 'sched_checkpoint.pt'))

            # swap params to ema values
            #optimizer.swap_ema()

            # evaluate
            eval_loss = evaluate(model, test_dataloader, loss_fn, args)
            #print('Evaluate bits per dim: {:.3f}'.format(eval_loss.item() / (np.log(2) * np.prod(args.image_dims))))
            print('Evaluate bits per dim: {:.3f}'.format(eval_loss / (np.log(2) * np.prod(args.image_dims))))
            #writer.add_scalar('eval_bits_per_dim', eval_loss.item() / (np.log(2) * np.prod(args.image_dims)), args.step)

            # generate
            samples, original = generate(model, test_dataloader, generate_fn, args)
            #writer.add_image('samples', samples, args.step)
            
            for z in range(3):
                save_image(samples[z], os.path.join(args.output_dir, 'generation_sample_step_{}_{}.png'.format(args.step, z+1)))
            save_image(original,  os.path.join(args.output_dir, 'generation_sample_step_{}_{}.png'.format(args.step, 'T')))
            
            # restore params to gradient optimized
            #optimizer.swap_ema()

# --------------------
# Main
# --------------------

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    #os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'#,8,9'#,1,2,3'
    os.environ['CUDA_VISIBLE_DEVICES']= '0,1,2,3,4,5,6,7'
    args = Parameters()
    args.output_dir = os.path.dirname(args.restore_file) if args.restore_file else \
                        os.path.join('results', args.model, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
    writer = SummaryWriter(log_dir = args.output_dir)

    # save config
    #if not os.path.exists(os.path.join(args.output_dir, 'config.json')): save_json(args.__dict__, 'config', args)
    #writer.add_text('config', str(args.__dict__))
    #pprint.pprint(args.__dict__)

    args.device = torch.device('cuda:{}'.format(args.cuda) if args.cuda is not None and torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dataloader, test_dataloader = fetch_dataloaders(args)

    if args.model=='pixelcnn':
        print('Using a plain PixelCNN')
        import pixelcnn
        #from pixelcnn import PixelCNN as pixelcnn
        model = pixelcnn.PixelCNN(args.image_dims, args.n_bits, args.n_channels, args.n_out_conv_channels, args.kernel_size,
                                  args.n_res_layers, args.n_cond_classes, args.norm_layer).to(args.device)
        # images need to be deprocessed to [0, 2**n_bits) for loss fn
        loss_fn = lambda scores, targets, n_bits: pixelcnn.loss_fn(scores, deprocess(targets, n_bits))
        # multinomial sampling needs to be processed to [-1,1] at generation
        generate_fn = partial(pixelcnn.generate_fn, preprocess_fn=preprocess, n_bits=args.n_bits)
        #optimizer = RMSprop(model.parameters(), lr=args.lr, polyak=args.polyak)
        optimizer = RMSprop(model.parameters(), lr=args.lr)
        scheduler = None
    elif args.model=='pixelcnnpp':
        print('Using PixelCNN++')
        import pixelcnnpp
        model = pixelcnnpp.PixelCNNpp(args.image_dims, args.n_channels, args.n_res_layers, args.n_logistic_mix,
                                      args.n_cond_classes).to(args.device)
        model = torch.nn.DataParallel(model).to(args.device)
        loss_fn = pixelcnnpp.loss_fn
        generate_fn = pixelcnnpp.generate_fn
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    elif args.model=='pixelsnail':
        print('Using PixelSnail')
        import pixelsnail, pixelcnnpp
        model = pixelsnail.PixelSNAIL(args.image_dims, args.n_channels, args.n_res_layers, args.attn_n_layers, args.attn_nh, 
                args.attn_dq, args.attn_dv, args.attn_drop_rate, args.n_logistic_mix, args.n_cond_classes).to(args.device)
        loss_fn = pixelcnnpp.loss_fn
        generate_fn = pixelcnnpp.generate_fn
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995), eps=1e-5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    else:
        print('No model selected')
        

    #    print(model)
    print('Model parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

    if args.restore_file:
        model_checkpoint = torch.load(args.restore_file, map_location=args.device)
        model.load_state_dict(model_checkpoint['state_dict'])
        optimizer.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/optim_checkpoint.pt', map_location=args.device))
        if scheduler:
            scheduler.load_state_dict(torch.load(os.path.dirname(args.restore_file) + '/sched_checkpoint.pt', map_location=args.device))
        args.start_epoch = model_checkpoint['epoch'] + 1
        args.step = model_checkpoint['global_step']

    if args.train:
        train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, scheduler, loss_fn, generate_fn, writer, args)

    if args.evaluate:
        #if args.step > 0: optimizer.swap_ema()
        eval_loss = evaluate(model, test_dataloader, loss_fn, args)
        print('Evaluate bits per dim: {:.3f}'.format(eval_loss / (np.log(2) * np.prod(args.image_dims))))
        #if args.step > 0: optimizer.swap_ema()

    if args.generate:
        #if args.step > 0: optimizer.swap_ema()
        samples = generate(model, generate_fn, args)
        writer.add_image('samples', samples, args.step)
        save_image(samples, os.path.join(args.output_dir, 'generation_sample_step_{}.png'.format(args.step)))
        #if args.step > 0: optimizer.swap_ema()








