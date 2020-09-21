import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Function
import cv2
import numpy as np
from torch.utils.cpp_extension import load
import os

base_path = os.getcwd()

cd = load(name="cd_condition_bbox",
          sources=[
              os.path.join(base_path, "layers/DefGrid/check_condition_lattice_bbox/check_condition_lattice_back2.cu"),
              os.path.join(base_path, "layers/DefGrid/check_condition_lattice_bbox/check_condition_lattice_for2.cu"),
              os.path.join(base_path, "layers/DefGrid/check_condition_lattice_bbox/check_condition_lattice.cpp")],
          verbose=True)

import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################
eps = 1e-10
debug = False


############################################3
# Inherit from Function
class TriRender2D(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, grid_bxkx3x2, img_pos_bxnx2, grid_mask=None):
        n_batch = grid_bxkx3x2.shape[0]
        n_pixel = img_pos_bxnx2.shape[1]
        # Initialize condition with all negative at first
        condition_bxnx1 = -torch.ones(n_batch, n_pixel, 1).float().to(grid_bxkx3x2.device)
        top_left, _ = torch.min(grid_bxkx3x2, dim=2)
        top_left = top_left.unsqueeze(2)
        bottom_right, _ = torch.max(grid_bxkx3x2, dim=2)
        bottom_right = bottom_right.unsqueeze(2)
        bbox_bxkx2x2 = torch.cat([top_left, bottom_right], dim=2)
        cd.forward(grid_bxkx3x2, img_pos_bxnx2, condition_bxnx1, bbox_bxkx2x2)

        return condition_bxnx1

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, condition_bxnx1):
        return None, None, None


###############################################################
check_condition_f_bbox = TriRender2D.apply


