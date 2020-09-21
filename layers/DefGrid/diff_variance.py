import matplotlib
matplotlib.use('agg')
import torch
import torch.nn as nn
import numpy as np
import cv2
import skimage
from layers.DefGrid.mean_feature.mean_feature import get_grid_mean_feature
from layers.DefGrid.variance_function_atom.line_distance_func_topk.utils import variance_f_in_one_atom_topk as line_variance_topk
from layers.DefGrid.variance_function_atom.line_distance_func_parallel.utils import variance_f_in_one_atom_parallel as line_variance_parallel
from layers.DefGrid.check_condition_lattice_bbox.utils import check_condition_f_bbox
EPS = 1e-10

class LatticeVariance(nn.Module):
    def __init__(self, output_row, output_column,  device, sigma=0.001,
                add_seg=False, mask_coef=0.3):

        super(LatticeVariance, self).__init__()

        self.output_row = output_row
        self.output_column = output_column
        self.sigma = torch.zeros(1).cuda()
        self.sigma[0] = sigma
        self.device = device

        # Construct Image's Pixel position
        output_x_pos = np.zeros((output_row, output_column))
        output_y_pos = np.zeros((output_row, output_column))
        for i in range(output_row):
            output_x_pos[i] = np.arange(output_column, dtype=np.float) / output_column
        for i in range(output_column):
            output_y_pos[:, i] = np.arange(output_row, dtype=np.float) / output_row

        # Each pixel lies in the center
        output_x_pos += 1.0 / output_column * 0.5
        output_y_pos += 1.0 / output_row * 0.5

        output_x_pos = torch.from_numpy(output_x_pos).float().to(device).unsqueeze(-1)
        output_y_pos = torch.from_numpy(output_y_pos).float().to(device).unsqueeze(-1)
        output_pos = torch.cat((output_x_pos, output_y_pos), -1)       
        self.register_buffer('output_pos', output_pos) 

        self.add_seg = add_seg
        self.mask_coef = mask_coef
    
    def forward(self, grid_pos=None, img_fea=None, base_triangle2point=None, base_area_mask=None, base_triangle_mask=None,
                area_normalize=(20, 20), semantic_mask=None, inference=False, img_pos=None, grid_size=None, grid_mask=None):

        #grid_pos: b x grid_num x 2
        n_batch = grid_pos.shape[0]
        gather_input = grid_pos.unsqueeze(2).expand(n_batch, grid_pos.shape[1], 3, 2)
        gather_index = base_triangle2point.unsqueeze(-1).expand(n_batch, base_triangle2point.shape[1], 3, 2).long()
        lattice = torch.gather(input=gather_input, dim=1, index=gather_index)

        def local_check_f(query):
            return check_condition_f_bbox(lattice, query)

        # import ipdb
        # ipdb.set_trace()
        lab_img_fea = img_fea
        if self.add_seg:
            assert not (semantic_mask is None)
            lab_img_fea = torch.cat((lab_img_fea, semantic_mask * self.mask_coef), dim=-1)

        variance_bxp, reconstruct_fea, condition = self.variance(lattice, lab_img_fea, inference=inference,
                                                                img_pos=img_pos,
                                                                grid_size=grid_size)
        ret = {'condition': condition}
        ret['check_condition'] = local_check_f
        ret['grid_lattice'] = lattice

        if inference:    
            return ret

        area_variance = self.area_variance(lattice, base_area_mask, area_normalize)
        
        ret['variance'] = variance_bxp.mean(dim=-1)
        ret['area_variance'] = area_variance

        reconstruct_loss = self.reconstruct(reconstruct_fea, lab_img_fea)
        ret['reconstruct_loss'] = reconstruct_loss
        ret['reconstruct_img'] = reconstruct_fea

        return ret

    def reconstruct(self, reconstruct_fea, img_fea):
        batch_size, h, w, c = img_fea.shape
        reconstruct_fea = reconstruct_fea.reshape(batch_size, h, w, c)
        reconstruct_loss = torch.abs(reconstruct_fea - img_fea).reshape(batch_size, -1).mean(dim=-1)
        return reconstruct_loss

    def variance(self, lattice, img_fea, inference, img_pos=None, grid_size=None):
        grid_num = lattice.shape[1]
        n_batch = lattice.shape[0]
        if img_pos is None:
            img_pos = self.output_pos.unsqueeze(0).expand(n_batch, self.output_row, self.output_column, 2)

        img_pos = img_pos.view(n_batch, -1, 2)
        img_pos = img_pos.contiguous()

        condition = check_condition_f_bbox(lattice, img_pos)

        if inference:
            return None, None, condition

        # Expect img_fea to have shape Bxpxd and condition to have shape Bxp
        img_fea = img_fea.reshape(n_batch, -1, img_fea.shape[-1])
        grid_fea, _ = get_grid_mean_feature(img_fea, condition.squeeze(-1), grid_num)
        grid_fea = grid_fea.detach()

        ######################################################################################
        # Get variance inside grid
        max_grid_size = max(grid_size[0] - 1, grid_size[1] - 1)
        sigma = 0.001 * 20 / max_grid_size # then there is no need to adjust
        new_sigma = torch.zeros(1, dtype=torch.float, device=lattice.device)
        new_sigma[0] = sigma

        # variance_bxp, reconstruct_img = line_variance_parallel(img_fea, grid_fea, lattice, img_pos, new_sigma, condition.squeeze())
        variance_bxp, reconstruct_img = line_variance_topk(img_fea, grid_fea, lattice, img_pos, new_sigma)

        return variance_bxp, reconstruct_img, condition

    def convertrgb2lab(self, batch_rgb):
        n_batch = batch_rgb.shape[0]
        batch_rgb_np = batch_rgb.cpu().numpy()
        batch_rgb_np = batch_rgb_np.astype(np.uint8)
        lab_list = []
        for i in range(n_batch):
            rgb = batch_rgb_np[i]
            lab = skimage.color.rgb2lab(rgb)
            lab_list.append(lab)
        lab = np.stack(lab_list, axis=0)
        return torch.from_numpy(lab).float().to(batch_rgb.device)

    def area_variance(self, lattice, base_area_mask, area_normalize=(20, 20)):
        tmp_lattice = lattice * torch.tensor([area_normalize[1], area_normalize[0]]).float().to(lattice.device).reshape(1, 1, 1, 2)
        A = tmp_lattice[:, :, 0, :]
        B = tmp_lattice[:, :, 1, :]
        C = tmp_lattice[:, :, 2, :]
        area1 = (A[..., 1] + B[..., 1]) * (B[..., 0] - A[..., 0]) / 2
        area2 = (B[..., 1] + C[..., 1]) * (C[..., 0] - B[..., 0]) / 2
        area3 = (C[..., 1] + A[..., 1]) * (A[..., 0] - C[..., 0]) / 2
        area = area1 + area2 + area3
        # we need to mask out some boundary lattice, which in theory is twice the innner triangle
        area = area.view(area.shape[0], -1)
        area = area * base_area_mask
        var_area = torch.var(area, dim=-1)
        return var_area

if __name__ == '__main__':
    pass
