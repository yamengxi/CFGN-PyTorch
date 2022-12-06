import os
import os.path as osp

import torch


root_dir = '/home/yamengxi/EDSR-PyTorch/models'

for model_name in os.listdir(root_dir):
    state_dict = torch.load(osp.join(root_dir, model_name), map_location='cpu') # 加载原来的模型，在torch>=1.6时加载
    torch.save(state_dict, osp.join(root_dir, model_name.split('.')[0] + '_old.' + model_name.split('.')[-1]), _use_new_zipfile_serialization=False) # 不是zip 

