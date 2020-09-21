import argparse
import os
import shutil
import torch

def str2bool(s):
    lower = s.lower()
    assert(lower == 'true' or lower == 'false')
    return lower == 'true'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_pos_layer', type=int, default=5)
    parser.add_argument('--update_last', type=str2bool, default=True)
    parser.add_argument('--grid_num', type=int, default=800)

    #epoch and step
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--print_freq', type=int, default=400)
    parser.add_argument('--sample_freq', type=int, default=20)

    #size
    parser.add_argument('--resolution', type=int, nargs='+', default=[224, 224])
    parser.add_argument('--state_dim', type=int, default=128)
    parser.add_argument('--grid_size', type=int, nargs='+', default=[20, 20])
    parser.add_argument('--feature_channel_num', type=int, default=128)
    parser.add_argument('--deform_layer_num', type=int, default=8)
    parser.add_argument('--mlp_expansion', type=int, default=4)
    parser.add_argument('--concat_channels', type=int, default=64)
    parser.add_argument('--final_dim', type=int, default=320)


    #training type
    parser.add_argument('--grid_type', type=str, default='dense_quad', choices=['lattice', 'quad', 'dense_quad', 'quad_angle'])
    parser.add_argument('--add_mask_variance', type=str2bool, default=True)
    parser.add_argument('--dextr_annotation', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='processed_cityscapes', choices=['processed_pascal', 'processed_cityscapes', 'full_pascal', 'full_cihp', 'processed_cityscapes_ade_mix', 'full_cityscapes', 'processed_cityscapes_ade'])
    parser.add_argument('--encoder_backbone', type=str, default='deeplab', choices=['deeplab', 'affinity_net', 'simplenn'])

    #learning parameter
    parser.add_argument('--lr_decay', type=str2bool, default=True)      
    parser.add_argument('--lr_decay_step', type=int, nargs='+', default=[8, 14])
    parser.add_argument('--grad_clip', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--mask_coef', type=float, default=0.3)
    parser.add_argument('--w_variance', type=float, default=1.0)
    parser.add_argument('--w_area', type=float, default=0.01)
    parser.add_argument('--w_laplacian', type=float, default=0.01)
    parser.add_argument('--w_reconstruct_loss', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.001)

    #transform
    parser.add_argument('--relax_crop', type=int, default=10)
    parser.add_argument('--zero_pad_crop', type=str2bool, default=True)
    parser.add_argument('--extreme_point', type=str2bool, default=False)

    #others
    parser.add_argument('--load', type=str2bool, default=False)
    parser.add_argument('--load_version', type=str, default='tmp')
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--debug', type=str2bool, default=False)    
    parser.add_argument('--debug_num', type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--version', type=str, default='debug')
    parser.add_argument('--debug_index', type=int, default=1)
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--val_debug', type=str2bool, default=False)
    parser.add_argument('--restart', type=str2bool, default=False)
    parser.add_argument('--use_slic', type=str2bool, default=False)
    args = parser.parse_args()
        
    return args
