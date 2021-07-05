# This is to save model
# 1. load model
# 2. save model!

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.utils.config import parse_train_file
from packnet_sfm.utils.horovod import hvd_init, rank
import torch

ckpt_path = '/home/seok436/data/md_model/default_config-train_nextchipDB-2021.06.23-22h46m00s_resnet_kitti_pt/epoch=20_-loss=0.000.ckpt'
saved_state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
torch.save(saved_state_dict, '/home/seok436/monodepth.pth')
print('load done')
