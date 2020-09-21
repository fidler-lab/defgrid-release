import torch
import torch.autograd
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os

base_path = os.getcwd()

line_variance_topk = load(name="line_variance_topk",
            sources = [os.path.join(base_path, "layers/DefGrid/variance_function_atom/line_distance_func_topk/variance_line_distance_for.cu"),
                       os.path.join(base_path, "layers/DefGrid/variance_function_atom/line_distance_func_topk/variance_line_distance_back.cu"),
                       os.path.join(base_path, "layers/DefGrid/variance_function_atom/line_distance_func_topk/variance_line_distance.cpp")],
            verbose=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################
eps = 1e-8
debug = False

############################################3
# Inherit from Function
class VarianceFunc(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, img_fea_bxnxd, grid_fea_bxkxd, grid_bxkx3x2, img_pos_bxnx2, sigma, eps=1e-5, topk=50):
        # import ipdb
        # ipdb.set_trace()
        # condition is which pixel in which grid
        n_batch = grid_bxkx3x2.shape[0]
        n_pixel = img_pos_bxnx2.shape[1]
        n_grid = grid_fea_bxkxd.shape[1]

        n_pixel_per_run = (1024 * 1024 * 128 / n_grid) + 1 # the buffer size is half GB
        n_img_split = int(n_pixel / n_pixel_per_run) + 1
        n_pixel_per_run = int(n_pixel / n_img_split) + n_img_split - 1

        # Reconstruct the original image
        img_fea_bxnxd = img_fea_bxnxd.contiguous()
        grid_fea_bxkxd = grid_fea_bxkxd.contiguous()
        if topk > n_grid:
            topk = n_grid - 1

        reconstruct_img = torch.zeros(n_batch, n_pixel, grid_fea_bxkxd.shape[-1], device=grid_bxkx3x2.device, dtype=torch.float)
        variance_bxn = torch.zeros(n_batch, n_pixel, device=grid_bxkx3x2.device, dtype=torch.float)
        top_k_grid = torch.zeros(n_batch, n_pixel, topk, device=grid_bxkx3x2.device, dtype=torch.long)
        buffer_bxnxk = torch.zeros(n_batch, n_pixel, topk, device=grid_bxkx3x2.device, dtype=torch.float)
        grid_center = torch.mean(grid_bxkx3x2, dim=2)
        # calculate the topk matrix
        for i in range(n_batch):
            tmp_grid_pos = grid_center[i].unsqueeze(0)
            for j in range(n_img_split - 1): # all pixels before the last batch
                tmp_img_pos = img_pos_bxnx2[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run].unsqueeze(1)
                dist = torch.abs(tmp_img_pos - tmp_grid_pos).sum(dim=-1)
                _, tmp_topk = torch.topk(dist, k=topk, dim=1, sorted=False, largest=False)
                top_k_grid[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run] = tmp_topk

            # last batch
            j = n_img_split - 1
            tmp_img_pos = img_pos_bxnx2[i][j * n_pixel_per_run:].unsqueeze(1)
            dist = torch.abs(tmp_img_pos - tmp_grid_pos).sum(dim=-1)
            _, tmp_topk = torch.topk(dist, k=topk, dim=1, sorted=False, largest=False)
            top_k_grid[i][j * n_pixel_per_run:] = tmp_topk

        top_k_grid = top_k_grid.float()
        line_variance_topk.forward(img_fea_bxnxd, grid_fea_bxkxd, grid_bxkx3x2,
                   img_pos_bxnx2, variance_bxn, sigma, reconstruct_img,
                   top_k_grid, buffer_bxnxk)
        ctx.save_for_backward(img_fea_bxnxd, grid_fea_bxkxd, grid_bxkx3x2, img_pos_bxnx2, top_k_grid, sigma)
        return variance_bxn, reconstruct_img

    @staticmethod
    def backward(ctx, dldvariance_bxn, dldreconstruct_img_bxnxd):
        img_fea_bxnxd, grid_fea_bxkxd, grid_bxkx3x2, img_pos_bxnx2, top_k_grid, sigma = ctx.saved_tensors
        n_batch = img_fea_bxnxd.shape[0]
        n_grid = grid_bxkx3x2.shape[1]
        n_pixel = img_fea_bxnxd.shape[1]
        topk = top_k_grid.shape[2]
        dldvariance_bxn = dldvariance_bxn.contiguous()
        dldreconstruct_img_bxnxd = dldreconstruct_img_bxnxd.contiguous()

        dldgrid_bxkx3x2 = torch.zeros(n_batch, n_grid, 3, 2, device=grid_bxkx3x2.device, dtype=torch.float)
        buffer_bxnxk = torch.zeros(n_batch, n_pixel, topk, device=grid_bxkx3x2.device, dtype=torch.float)
        line_variance_topk.backward(dldvariance_bxn, img_fea_bxnxd, grid_fea_bxkxd,
                    grid_bxkx3x2,
                    img_pos_bxnx2, sigma[0].item(), dldreconstruct_img_bxnxd, top_k_grid,
                    buffer_bxnxk, dldgrid_bxkx3x2)

        return None, None, dldgrid_bxkx3x2, None, None, None, None


###############################################################
variance_f_in_one_atom_topk = VarianceFunc.apply
