import torch
import torch.nn as nn
import dataloaders.helpers as helpers
import torch.nn.functional as F
import numpy as np

from Models.GNN.GCN import GCN

from copy import deepcopy


class EncoderHead(nn.Module):
    def __init__(self, input_feature_channel_num=128):
        super(EncoderHead, self).__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(input_feature_channel_num, input_feature_channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_feature_channel_num),
            nn.ReLU(),
            nn.Conv2d(input_feature_channel_num, input_feature_channel_num, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_feature_channel_num),
            nn.ReLU(),
            nn.Conv2d(input_feature_channel_num, input_feature_channel_num, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        return self.predictor(x)


class DeformGNN(nn.Module):
    def __init__(self,
                 state_dim=256,
                 feature_channel_num=128,
                 out_dim=2,
                 layer_num=8,
                 scale_pos=False,
                 use_att=False
                 ):

        super(DeformGNN, self).__init__()
        self.state_dim = state_dim
        self.feature_channel_num = feature_channel_num
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.scale_pos = scale_pos
        self.use_att = use_att
        self.gnn = GCN(state_dim=self.state_dim, feature_dim=self.feature_channel_num + 2, out_dim=self.out_dim, layer_num=self.layer_num)


        print ('DeformGCN Initialization')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # m.weight.data.normal_(0.0, 0.00002)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)
        print('DeformGCN End Initialization')

    def forward(self, features, base_point, base_normalized_point_adjacent, base_point_mask, old_point_mask=None,
                update_last=False, scale=0.1):
        """
        pred_polys: in scale [0,1]
        """
        out_dict = {}
        shape = features.shape
        hw = shape[-2:]
        tmp_features = features.reshape(shape[0], shape[1], -1)
        tmp_features = tmp_features.permute(0, 2, 1).contiguous()
        cnn_feature = self.interpolated_sum([tmp_features], base_point, [hw])

        input_feature = torch.cat((cnn_feature, base_point), 2)

        gcn_pred = self.gnn.forward(input_feature, base_normalized_point_adjacent)
        # import ipdb
        # ipdb.set_trace()
        # gcn_pred = (torch.sigmoid(gcn_pred) - 0.5) * scale * 2

        enforced_gcn_pred = gcn_pred
        gcn_pred_poly = base_point + enforced_gcn_pred[:, :, :2] * base_point_mask.squeeze(1)


        laplacian_coord_1 = base_point - torch.bmm(base_normalized_point_adjacent, base_point)
        laplacian_coord_2 = gcn_pred_poly - torch.bmm(base_normalized_point_adjacent, gcn_pred_poly)
        laplacian_energy = ((laplacian_coord_2 - laplacian_coord_1) ** 2 + 1e-10).sum(-1).sqrt()
        laplacian_energy = laplacian_energy.mean(dim=-1)
        out_dict['laplacian_energy'] = laplacian_energy

        out_dict['pred_points'] = gcn_pred_poly
        out_dict['gcn_pred_points'] = gcn_pred
        return out_dict

    def interpolated_sum(self, cnns, coords, grids, grid_multiplier=None):
        X = coords[:, :, 0]
        Y = coords[:, :, 1]

        cnn_outs = []

        for i in range(len(grids)):
            grid = grids[i]

            #x is the horizontal coordinate
            if grid_multiplier is None:
                Xs = X * grid[1]
            else:
                Xs = X * grid_multiplier[i][1]
            X0 = torch.floor(Xs)
            X1 = X0 + 1

            if grid_multiplier is None:
                Ys = Y * grid[0]
            else:
                Ys = Y * grid_multiplier[i][1]
            Y0 = torch.floor(Ys)
            Y1 = Y0 + 1

            w_00 = (X1 - Xs) * (Y1 - Ys)
            w_01 = (X1 - Xs) * (Ys - Y0)
            w_10 = (Xs - X0) * (Y1 - Ys)
            w_11 = (Xs - X0) * (Ys - Y0)

            X0 = torch.clamp(X0, 0, grid[1]-1)
            X1 = torch.clamp(X1, 0, grid[1]-1)
            Y0 = torch.clamp(Y0, 0, grid[0]-1)
            Y1 = torch.clamp(Y1, 0, grid[0]-1)

            N1_id = X0 + Y0 * grid[1]
            N2_id = X0 + Y1 * grid[1]
            N3_id = X1 + Y0 * grid[1]
            N4_id = X1 + Y1 * grid[1]

            M_00 = helpers.gather_feature(N1_id, cnns[i])
            M_01 = helpers.gather_feature(N2_id, cnns[i])
            M_10 = helpers.gather_feature(N3_id, cnns[i])
            M_11 = helpers.gather_feature(N4_id, cnns[i])

            cnn_out = w_00.unsqueeze(2) * M_00 + \
                      w_01.unsqueeze(2) * M_01 + \
                      w_10.unsqueeze(2) * M_10 + \
                      w_11.unsqueeze(2) * M_11

            cnn_outs.append(cnn_out)

        concat_features = torch.cat(cnn_outs, dim=2)

        return concat_features
