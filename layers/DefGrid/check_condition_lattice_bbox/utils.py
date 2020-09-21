import torch
import torch.autograd
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os

base_path = os.getcwd()

check_condition = load(name="check_condition_bbox",
          sources=[
              os.path.join(base_path, "layers/DefGrid/check_condition_lattice_bbox/check_condition_lattice_for2.cu"),
              os.path.join(base_path, "layers/DefGrid/check_condition_lattice_bbox/check_condition_lattice.cpp")],
          verbose=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################3
class CheckCondition(Function):

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
        check_condition.forward(grid_bxkx3x2, img_pos_bxnx2, condition_bxnx1, bbox_bxkx2x2)
        return condition_bxnx1

    @staticmethod
    def backward(ctx, condition_bxnx1):
        return None, None, None


###############################################################
check_condition_f_bbox = CheckCondition.apply


