import torch

class Parameters:
    def __init__(self):
        self.model='pixelcnnpp'
        #self.n_channels=128             #'Number of channels for gated residual convolutional layers.')
        self.n_out_conv_channels=1024     # 1024 #'Number of channels for outer 1x1 convolutional layers.')
        #self.n_res_layers=12            #'Number of Gated Residual Blocks.')
        self.kernel_size=5              #'Kernel size for the gated residual convolutional blocks.')
        self.norm_layer=True            #'Add a normalization layer in every Gated Residual Blocks.')
        self.device = torch.device('cuda')
        
        # pixelcnn++ args
        #self.n_channels=128     #'Number of channels for residual blocks.')
        #self.n_res_layers=5     #'Number of residual blocks at each stage.')
        #self.n_logistic_mix=10  #'Number of of mixture components for logistics output.')
        
        # pixelsnail args
        self.n_channels=256     # 256 #'Number of channels for residual blocks.')
        self.n_res_layers=5     # 5 #'Number of residual blocks in each attention layer.')
        self.attn_n_layers=12   # 12 #'Number of attention layers.')
        self.attn_nh=1          # 1'Number of attention heads.')
        self.attn_dq=16         # 16 #'Size of attention queries and keys.')
        self.attn_dv=128        # 128 #'Size of attention values.')
        self.attn_drop_rate=0   #'Dropout rate on attention logits.')
        self.n_logistic_mix=10  #'Number of of mixture components for logistics output.')
        
        # action@
        self.train=True         #'Train model.')
        self.evaluate=False      #'Evaluate model.')
        self.generate=False      #'Generate samples from a model.')
        self.output_dir=r'C:\Users\W.Rogers\PixelCNN\ii\checkpoints'  #'Path to model to restore.')
        self.restore_file=False
        self.seed=0             #'Random seed to use.')
        self.cuda='0,1,2,3,4,5,6,7'           #'Which cuda device to use.')
        
        # data params
        self.dataset='phantom_control_seen'
        self.n_cond_classes=None    #'Number of classes for class conditional model.')
        self.n_bits=4               #'Number of bits of input data.')
        self.image_dims=(3,64,64) #'Dimensions of the input data.')
        self.output_dims=(3,64,64) #'Dimensions of the output data')
        self.data_path=r''       #help='Location of datasets.')
        self.mini_data=False
        
        # training param
        self.lr=3e-5            #'Learning rate.')
        self.lr_decay=0.999995  #'Learning rate decay, applied every step of the optimization.')
        self.polyak=0.9995      #'Polyak decay for parameter exponential moving average.')
        self.batch_size=32      #'Training batch size.')
        self.n_epochs=8192        #'Number of epochs to train.')
        self.step=0             #'Current step of training (number of minibatches processed).')
        self.start_epoch=0      #'Starting epoch (for logging; to be overwritten when restoring file.')
        self.log_interval=2048    #'How often to show loss statistics and save samples.')
        self.eval_interval=2048   #'How often to evaluate and save samples.')
        #self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR
        self.scheduler = None
        
        # generation param
        self.n_samples=1        #'Number of samples to generate.')

        # initializers
        initializers = [None,
                        torch.nn.init.uniform_,         #1
                        torch.nn.init.normal_,          #2
                        torch.nn.init.constant_,        #3
                        torch.nn.init.ones_,            #4
                        torch.nn.init.zeros_,           #5
                        torch.nn.init.dirac_,           #6
                        torch.nn.init.xavier_uniform_,  #7
                        torch.nn.init.xavier_normal_,   #8
                        torch.nn.init.kaiming_uniform_, #9
                        torch.nn.init.kaiming_normal_,  #10
                        torch.nn.init.orthogonal_]      #11
        
        self.init=initializers[8]








