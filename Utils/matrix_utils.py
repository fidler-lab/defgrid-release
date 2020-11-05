import torch
from Utils import dense_quad
import numpy as np

class MatrixUtils:
    def __init__(self, batch_size, grid_size, grid_type, device):
        """
        Initialize a grid.

        Args:
            self: (todo): write your description
            batch_size: (int): write your description
            grid_size: (int): write your description
            grid_type: (str): write your description
            device: (todo): write your description
        """
        self.batch_size = batch_size
        self.grid_size = grid_size

        self.point_num = self.grid_size[0] * self.grid_size[1]
        self.grid_num = (self.grid_size[0] - 1) * (self.grid_size[1] - 1) * 2
        self.init_point = dense_quad.create_init_point(self.grid_size[0], self.grid_size[1])
        self.init_point_mask = dense_quad.create_init_point_mask(self.grid_size[0], self.grid_size[1])
        self.triangle2point, self.init_area_mask = dense_quad.create_initial_triangle(self.grid_size[0], self.grid_size[1])
        self.init_point_adjacent = dense_quad.create_init_point_adjacent(self.grid_size[0], self.grid_size[1])
        self.init_cell_adjacent = dense_quad.create_init_cell_adjacent(self.grid_size[0], self.grid_size[1])
        self.init_cell_adjacent_my = dense_quad.create_init_cell_adjacent_jun_faster(self.triangle2point)
        self.init_edge_ex2, self.init_edge_face_id, self.init_all_edge_ex2, = dense_quad.create_edge_idx(self.triangle2point.data.numpy(), self.init_point.shape[0])
        self.init_edge_adj_nxn = dense_quad.create_edge_adj(self.init_all_edge_ex2)
        self.init_edge_ex2 = torch.from_numpy(self.init_edge_ex2)
        self.init_edge_face_id = torch.from_numpy(self.init_edge_face_id)
        self.init_all_edge_ex2 = torch.from_numpy(self.init_all_edge_ex2)
        self.init_edge_adj_nxn = torch.from_numpy(self.init_edge_adj_nxn).float()

        # init_cell_adjacent_my = dense_quad.create_init_cell_adjacent_jun_faster(self.triangle2point)
        self.init_point = self.init_point.unsqueeze(0).expand(self.batch_size, self.init_point.shape[-2], self.init_point.shape[-1]).float().to(device)
        self.init_point_mask = self.init_point_mask.unsqueeze(0).expand(self.batch_size, self.init_point_mask.shape[-2], self.init_point_mask.shape[-1]).float().to(device)
        self.init_triangle2point = self.triangle2point.unsqueeze(0).repeat(self.batch_size, 1, 1).long().to(device)
        self.init_area_mask = self.init_area_mask.unsqueeze(0).repeat(self.batch_size, 1).float().to(device)
        self.init_point_adjacent = self.init_point_adjacent.unsqueeze(0).repeat(self.batch_size, 1, 1).float().to(device)
        self.init_cell_adjacent_my = self.init_cell_adjacent_my.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).float().to(device)
        self.init_normalized_point_adjacent = self.init_point_adjacent / torch.sum(self.init_point_adjacent, dim=2, keepdim=True).repeat(1, 1, self.init_point_adjacent.shape[-1])
        self.init_triangle_mask = torch.zeros(self.batch_size, self.init_area_mask.shape[1]).byte().to(device)


    def get_new_matrix(self, img_size, n_grid, device, batch_size=1, upsample=False, scale_pos=False, upsample_scale=2):
        """
        Return a new 2dense image.

        Args:
            self: (todo): write your description
            img_size: (int): write your description
            n_grid: (int): write your description
            device: (todo): write your description
            batch_size: (int): write your description
            upsample: (int): write your description
            scale_pos: (str): write your description
            upsample_scale: (str): write your description
        """
        img_height = img_size[0]
        img_width = img_size[1]
        # calculate the number of superpixel
        k = int(np.floor(n_grid / 2))
        k_w = int(np.floor(np.sqrt(k * img_width / img_height))) # number of cells
        k_h = int(np.floor(np.sqrt(k * img_height / img_width)))

        grid_size = [k_h + 1, k_w + 1] # number of vertices
        point_num = grid_size[0] * grid_size[1]
        grid_num = (grid_size[0] - 1) * (grid_size[1] - 1) * 2
        init_point = dense_quad.create_init_point(grid_size[0], grid_size[1], scale_pos=scale_pos, img_size=img_size)
        init_point_mask = dense_quad.create_init_point_mask(grid_size[0], grid_size[1])
        triangle2point, init_area_mask = dense_quad.create_initial_triangle(grid_size[0], grid_size[1])

        init_point_adjacent = dense_quad.create_init_point_adjacent(grid_size[0], grid_size[1])
        # init_cell_adjacent = dense_quad.create_init_cell_adjacent(grid_size[0], grid_size[1])
        init_cell_adjacent_my = dense_quad.create_init_cell_adjacent_jun_faster(triangle2point)
        init_point = init_point.unsqueeze(0).expand(batch_size, init_point.shape[-2], init_point.shape[-1]).float().to(device)
        init_point_mask = init_point_mask.unsqueeze(0).expand(batch_size, init_point_mask.shape[-2], init_point_mask.shape[-1]).float().to(device)
        init_triangle2point = triangle2point.unsqueeze(0).repeat(batch_size, 1, 1).long().to(device)
        init_area_mask = init_area_mask.unsqueeze(0).repeat(batch_size, 1).float().to(device)
        init_point_adjacent = init_point_adjacent.unsqueeze(0).repeat(batch_size, 1, 1).float().to(device)
        init_cell_adjacent_my = init_cell_adjacent_my.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(device)
        init_normalized_point_adjacent = init_point_adjacent / torch.sum(init_point_adjacent, dim=2, keepdim=True)

        return init_point, init_point_adjacent, init_normalized_point_adjacent, init_point_mask, init_triangle2point, \
                init_area_mask, point_num, grid_num, grid_size

    def get_img_pos(self, img_size, device):
        """
        Get the position of the image at the given device.

        Args:
            self: (todo): write your description
            img_size: (int): write your description
            device: (todo): write your description
        """
        output_row = img_size[0]
        output_column = img_size[1]

        output_x_pos = np.zeros((output_row, output_column))
        output_y_pos = np.zeros((output_row, output_column))

        for i in range(output_row):
            output_x_pos[i] = np.arange(output_column, dtype=np.float) / output_column
        for i in range(output_column):
            output_y_pos[:, i] = np.arange(output_row, dtype=np.float) / output_row
        # Each pixel corresponds to the center of the grid

        output_x_pos += 1.0 / output_column * 0.5
        output_y_pos += 1.0 / output_row * 0.5

        output_x_pos = torch.from_numpy(output_x_pos).float().to(device).unsqueeze(-1)
        output_y_pos = torch.from_numpy(output_y_pos).float().to(device).unsqueeze(-1)
        output_pos = torch.cat((output_x_pos, output_y_pos), -1)
        return output_pos


