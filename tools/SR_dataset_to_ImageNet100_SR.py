import os
import os.path as osp
import shutil

SR_dataset_root_dir = '/data2/yamengxi/CFGN/CFGN-PyTorch/experiment/final_result_dilation=3_num_G=32_num_TFC=3_CFGN_CFGM_v2+ACT_BIX2_F64R9_2023-01-27_21:30:56/ImageNet100'
ImageNet100_root_dir = '/data2/yamengxi/CFGN/datasets/ImageNet_100'

os.makedirs(osp.join(ImageNet100_root_dir, 'val_SR'), exist_ok=True)

for image_name in list(os.listdir(SR_dataset_root_dir)):
    os.makedirs(osp.join(ImageNet100_root_dir, 'val_SR', image_name.split('_')[1]), exist_ok=True)
    shutil.copy(osp.join(SR_dataset_root_dir, image_name), osp.join(ImageNet100_root_dir, 'val_SR', image_name.split('_')[1], image_name.split('_')[2] + '_' + image_name.split('_')[3] + '.png'))
