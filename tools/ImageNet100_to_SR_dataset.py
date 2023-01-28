import os
import os.path as osp

import cv2


ImageNet100_root_dir = '/data2/yamengxi/CFGN/datasets/ImageNet_100'
target_root_dir = '/data2/yamengxi/CFGN/datasets_for_EDSR-PyTorch/benchmark/ImageNet100'

os.makedirs(osp.join(target_root_dir, 'HR'), exist_ok=True)
os.makedirs(osp.join(target_root_dir, 'LR_bicubic', 'X2'), exist_ok=True)

for mid_dir in ['val']:
    for class_dir in list(os.listdir(osp.join(ImageNet100_root_dir, mid_dir))):
        print(osp.join(ImageNet100_root_dir, mid_dir, class_dir))
        for image_name in list(os.listdir(osp.join(ImageNet100_root_dir, mid_dir, class_dir))):
            img = cv2.imread(osp.join(ImageNet100_root_dir, mid_dir, class_dir, image_name), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(osp.join(target_root_dir, 'HR', f'{mid_dir}_{class_dir}_{image_name.split(".")[0]}.png'), img)
            img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(osp.join(target_root_dir, 'LR_bicubic', 'X2', f'{mid_dir}_{class_dir}_{image_name.split(".")[0]}x2.png'), img)
