from easydict import EasyDict
import torch

config = EasyDict()

# Normalization
config.means = (0.485, 0.456, 0.406)
config.stds = (0.229, 0.224, 0.225)

config.log_dir = "log"
config.exp_name = "synthtext"
config.device = "cuda"
config.save_dir = "weights"

config.display_freq = 100
config.viz = True
config.viz_freq = 1000
config.vis_dir = "viz"

config.log_freq = 1000
config.save_freq = 1

# dataloader jobs number
config.num_workers = 0

# batch_size
config.batch_size = 4

# training epoch number
config.max_epoch = 200

config.start_epoch = 0

# learning rate
config.lr = 1e-3

# using GPU
config.cuda = True

config.n_disk = 15

config.output_dir = 'output'

config.input_size = 512

# max polygon per image
config.max_annotation = 200

# max point per polygon
config.max_points = 20

# use hard examples (annotated as '#')
config.use_hard = True

# demo tr threshold
config.tr_thresh = 0.6

# demo tcl threshold
config.tcl_thresh = 0.4

# expand ratio in post processing
config.post_process_expand = 0.3

# merge joined text instance when predicting
config.post_process_merge = False

def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
