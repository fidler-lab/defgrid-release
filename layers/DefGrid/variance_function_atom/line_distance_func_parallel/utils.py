import torch
import torch.autograd
from torch.autograd import Function
from torch.utils.cpp_extension import load
import os

base_path = os.getcwd()

line_variance_parallel = load(name="line_variance_parallel",
            sources = [os.path.join(base_path, "layers/DefGrid/variance_function_atom/line_distance_func_parallel/variance_line_distance_for.cu"),
                       os.path.join(base_path, "layers/DefGrid/variance_function_atom/line_distance_func_parallel/variance_line_distance_back.cu"),
                       os.path.join(base_path, "layers/DefGrid/variance_function_atom/line_distance_func_parallel/variance_line_distance.cpp")],
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
    def forward(ctx, img_fea_bxnxd, grid_fea_bxkxd, grid_bxkx3x2, img_pos_bxnx2, sigma, condition, eps=1e-5):
        # import ipdb
        # ipdb.set_trace()
        # condition is which pixel in which grid
        n_batch = grid_bxkx3x2.shape[0]
        n_pixel = img_pos_bxnx2.shape[1]
        n_grid = grid_fea_bxkxd.shape[1]
        n_fea = grid_fea_bxkxd.shape[2]
        n_block = 1024
        n_pixel_per_run = (1024 * 1024 * 128 / n_grid) + 1 # the buffer size is half GB
        split_size = int(n_grid / n_block) + 1
        n_img_split = int(n_pixel / n_pixel_per_run) + 1
        n_pixel_per_run = int(n_pixel / n_img_split) + n_img_split - 1

        # Reconstruct the original image
        img_fea_bxnxd = img_fea_bxnxd.contiguous()
        grid_fea_bxkxd = grid_fea_bxkxd.contiguous()

        reconstruct_img = torch.zeros(n_batch, n_pixel, grid_fea_bxkxd.shape[-1], device=grid_bxkx3x2.device, dtype=torch.float)
        variance_bxn = torch.zeros(n_batch, n_pixel, device=grid_bxkx3x2.device, dtype=torch.float)
        # Buffer is used to save intermediate results, otherwise should apply memory in gpu.
        # we could also limit the size of buffer, to make it suitable for large image.
        buffer_1xnxk = torch.zeros(1, int(n_pixel / n_img_split) + n_img_split - 1, n_grid, device=grid_bxkx3x2.device, dtype=torch.float)
        buffer_1xn = torch.zeros(1, int(n_pixel / n_img_split) + n_img_split - 1, device=grid_bxkx3x2.device, dtype=torch.float)
        buffer_1xnx4 = torch.zeros(1, int(n_pixel / n_img_split) + n_img_split - 1, split_size, device=grid_bxkx3x2.device, dtype=torch.float)
        buffer_1xnxdx4 = torch.zeros(1, int(n_pixel / n_img_split) + n_img_split - 1, n_fea, split_size, device=grid_bxkx3x2.device, dtype=torch.float)
        #variance_bxk = torch.empty(n_batch, n_grid)

        for i in range(n_batch):
            buffer_1xnxk.zero_()
            buffer_1xn.zero_()
            buffer_1xnx4.zero_()
            buffer_1xnxdx4.zero_()
            for j in range(n_img_split - 1): # all pixels before the last batch
                line_variance_parallel.forward(img_fea_bxnxd[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run].unsqueeze(0),
                           grid_fea_bxkxd[i].unsqueeze(0),
                           grid_bxkx3x2[i].unsqueeze(0),
                            img_pos_bxnx2[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run].unsqueeze(0),
                           variance_bxn[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run].unsqueeze(0),
                           sigma,
                           reconstruct_img[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run].unsqueeze(0),
                           buffer_1xnxk, buffer_1xn, buffer_1xnx4, buffer_1xnxdx4, split_size)
                variance_bxn[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run] = buffer_1xnxk.sum(-1).squeeze()
            # last batch
            j = n_img_split - 1
            n_pixel_in_last = n_pixel - n_pixel_per_run * j
            line_variance_parallel.forward(img_fea_bxnxd[i][j * n_pixel_per_run:].unsqueeze(0),
                       grid_fea_bxkxd[i].unsqueeze(0),
                       grid_bxkx3x2[i].unsqueeze(0),
                       img_pos_bxnx2[i][j * n_pixel_per_run:].unsqueeze(0),
                       variance_bxn[i][j * n_pixel_per_run:].unsqueeze(0),
                       sigma,
                       reconstruct_img[i][j * n_pixel_per_run:].unsqueeze(0),
                       buffer_1xnxk[:,:n_pixel_in_last], buffer_1xn[:,:n_pixel_in_last], buffer_1xnx4[:,:n_pixel_in_last],
                       buffer_1xnxdx4[:,:n_pixel_in_last], split_size)
            variance_bxn[i][j * n_pixel_per_run:] = buffer_1xnxk[:,:n_pixel_in_last].sum(-1).squeeze()

        ctx.save_for_backward(img_fea_bxnxd, grid_fea_bxkxd, grid_bxkx3x2, img_pos_bxnx2, sigma)
        return variance_bxn, reconstruct_img

    @staticmethod
    def backward(ctx, dldvariance_bxn, dldreconstruct_img_bxnxd):
        img_fea_bxnxd, grid_fea_bxkxd, grid_bxkx3x2, img_pos_bxnx2, sigma = ctx.saved_tensors
        # all_start_t = time.time()
        n_batch = img_fea_bxnxd.shape[0]
        n_grid = grid_bxkx3x2.shape[1]
        n_pixel = img_fea_bxnxd.shape[1]
        dldvariance_bxn = dldvariance_bxn.contiguous()
        dldreconstruct_img_bxnxd = dldreconstruct_img_bxnxd.contiguous()

        n_block = 1024
        n_pixel_per_run = (1024 * 1024 * 128 / n_grid) + 1  # the buffer size is half GB
        split_size = int(n_grid / n_block) + 1
        n_img_split = int(n_pixel / n_pixel_per_run) + 1
        n_pixel_per_run = int(n_pixel / n_img_split) + n_img_split - 1


        dldgrid_bxkx3x2 = torch.zeros(n_batch, n_grid, 3, 2, device=grid_bxkx3x2.device, dtype=torch.float)
        # gradient_1xnxkx3x2 = torch.zeros(1, n_pixel, n_grid, 3, 2).float().to(grid_bxkx3x2.device)
        buffer_1xnxk = torch.zeros(1, int(n_pixel / n_img_split) + n_img_split - 1, n_grid, device=grid_bxkx3x2.device, dtype=torch.float)
        buffer_1xn = torch.zeros(1, int(n_pixel / n_img_split) + n_img_split - 1, device=grid_bxkx3x2.device, dtype=torch.float)
        buffer_1xnx4 = torch.zeros(1, int(n_pixel / n_img_split) + n_img_split - 1, split_size, device=grid_bxkx3x2.device, dtype=torch.float)
        # start_t = time.time()
        for i in range(n_batch):
            buffer_1xnxk.zero_()
            buffer_1xn.zero_()
            buffer_1xnx4.zero_()

            for j in range(n_img_split - 1):  # all pixels before the last batch
                line_variance_parallel.backward(dldvariance_bxn[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run].unsqueeze(0),
                            img_fea_bxnxd[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run].unsqueeze(0),
                            grid_fea_bxkxd[i].unsqueeze(0), grid_bxkx3x2[i].unsqueeze(0),
                            img_pos_bxnx2[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run].unsqueeze(0),
                            sigma[0].item(),
                            dldreconstruct_img_bxnxd[i][j * n_pixel_per_run: (j + 1) * n_pixel_per_run].unsqueeze(0),
                            buffer_1xnxk, dldgrid_bxkx3x2[i].unsqueeze(0),
                            buffer_1xn, buffer_1xnx4, split_size)
            # last batch
            j = n_img_split - 1
            n_pixel_in_last = n_pixel - n_pixel_per_run * j
            line_variance_parallel.backward(dldvariance_bxn[i][j * n_pixel_per_run:].unsqueeze(0),
                        img_fea_bxnxd[i][j * n_pixel_per_run:].unsqueeze(0),
                        grid_fea_bxkxd[i].unsqueeze(0), grid_bxkx3x2[i].unsqueeze(0),
                        img_pos_bxnx2[i][j * n_pixel_per_run:].unsqueeze(0),
                        sigma[0].item(),
                        dldreconstruct_img_bxnxd[i][j * n_pixel_per_run:].unsqueeze(0),
                        buffer_1xnxk[:,:n_pixel_in_last], dldgrid_bxkx3x2[i].unsqueeze(0),
                        buffer_1xn[:,:n_pixel_in_last], buffer_1xnx4[:,:n_pixel_in_last], split_size)
        # end_t = time.time()
        # print('backward time: %.5f' %(end_t - start_t))
        return None, None, dldgrid_bxkx3x2, None, None, None


###############################################################
variance_f_in_one_atom_parallel = VarianceFunc.apply



