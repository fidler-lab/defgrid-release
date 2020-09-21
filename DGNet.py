from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataloaders import helpers

from Models.GNN import superpixel_grid
from layers.DefGrid.diff_variance import LatticeVariance
from layers.DefGrid.mean_feature.mean_feature import get_grid_mean_feature
from Utils.matrix_utils import MatrixUtils
from lib.sync_bn.modules import BatchNorm2d


EPS = 1e-8


class DGNet(nn.Module):
    def __init__(self, args):
        super(DGNet, self).__init__()
        # args
        self.args = args
        self.grid_size = args.grid_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ''' resnet '''
        resnet = torchvision.models.resnet50(pretrained=True, norm_layer=BatchNorm2d)
        self.shallow_encoder = nn.Sequential(*list(resnet.children())[:5])
        self.layer2, self.layer3, self.layer4 = resnet.layer2, resnet.layer3, resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        ''' skip connection '''
        concat_channels = 128
        concat1 = nn.Conv2d(256, concat_channels, kernel_size=3, padding=1, bias=False)
        bn1 = BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)
        self.res1_concat = nn.Sequential(concat1, bn1, relu1)
        concat2 = nn.Conv2d(512, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)
        up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.res2_concat = nn.Sequential(concat2, bn2, relu2, up2)
        concat3 = nn.Conv2d(1024, concat_channels, kernel_size=3, padding=1, bias=False)
        bn3 = BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)
        up3 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.res3_concat = nn.Sequential(concat3, bn3, relu3, up3)
        concat4 = nn.Conv2d(2048, concat_channels, kernel_size=3, padding=1, bias=False)
        bn4 = BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)
        up4 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.res4_concat = nn.Sequential(concat4, bn4, relu4, up4)

        ''' classifier '''
        self.cls = nn.Sequential(
            nn.Conv2d(4 * concat_channels, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, 19, kernel_size=1)
        )
        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, 19, kernel_size=1)
        )

        self.modules_ori = [self.layer2, self.layer3, self.layer4]
        self.modules_new = [self.cls, self.aux, self.res1_concat, self.res2_concat, self.res3_concat, self.res4_concat]
        for block in self.modules_new:
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        ''' grid '''
        out_feature_channel_num = 256
        self.deformer_conv1 = nn.Sequential(
            nn.Conv2d(out_feature_channel_num, out_feature_channel_num, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(out_feature_channel_num),
            nn.ReLU(inplace=True)
        )
        self.deformer = superpixel_grid.DeformGNN(state_dim=self.args.state_dim,
                                                  feature_channel_num=out_feature_channel_num, out_dim=2,
                                                  layer_num=self.args.deform_layer_num
                                                  )

        self.matrix = MatrixUtils(1, self.args.grid_size, self.args.grid_type, device=torch.device("cuda"))
        self.lattice = LatticeVariance(self.args.train_h, self.args.train_w, sigma=self.args.sigma,
                                       device=self.device, add_seg=self.args.add_mask_variance, mask_coef=self.args.mask_coef)
        self.feature_lattice = LatticeVariance(self.args.train_h // 4, self.args.train_w // 4, sigma=self.args.sigma, device=self.device)

        self.modules_dg = [self.shallow_encoder, self.deformer_conv1, self.deformer]

    def forward(self, net_input, bifilter_crop_image=None, crop_gt=None, inference=False, semantic_edge=None):
        # init grid utils
        n_batch = net_input.shape[0]
        base_point = self.matrix.init_point.expand(n_batch, -1, -1).cuda()
        base_normalized_point_adjacent = torch.stack([self.matrix.init_normalized_point_adjacent[0]] * n_batch).cuda()
        base_point_mask = self.matrix.init_point_mask.expand(n_batch, -1, -1).cuda()
        base_triangle2point = self.matrix.init_triangle2point.expand(n_batch, -1, -1).cuda()
        base_area_mask = self.matrix.init_area_mask.expand(n_batch, -1).cuda()
        base_triangle_mask = self.matrix.init_triangle_mask.expand(n_batch, -1).cuda()

        # shallow forward
        layer1_f = self.shallow_encoder(net_input)

        # Grid Deformation
        deformer_feature = self.deformer_conv1(layer1_f)
        output = self.deformer(deformer_feature, base_point, base_normalized_point_adjacent, base_point_mask)
        pred_points = output['pred_points']

        # compute grid info
        if self.args.add_mask_variance:
            tmp_gt_mask = deepcopy(crop_gt)
            gt_mask = helpers.gtmask2onehot(tmp_gt_mask).permute(0, 2, 3, 1)
            gt_mask = gt_mask[:,:,:,:19]        # remove ignore
            superpixel_ret = self.lattice(grid_pos=pred_points,
                                          img_fea=bifilter_crop_image[:, :3, ...].permute(0, 2, 3, 1),
                                          base_triangle2point=base_triangle2point, base_area_mask=base_area_mask,
                                          base_triangle_mask=base_triangle_mask,
                                          area_normalize=(self.args.grid_size[0], self.args.grid_size[1]),
                                          grid_size=self.grid_size,
                                          semantic_mask=gt_mask, inference=inference)
        else:
            superpixel_ret = self.lattice(grid_pos=pred_points,
                                          img_fea=bifilter_crop_image[:, :3, ...].permute(0, 2, 3, 1),
                                          base_triangle2point=base_triangle2point, base_area_mask=base_area_mask,
                                          base_triangle_mask=base_triangle_mask,
                                          area_normalize=(self.args.grid_size[0], self.args.grid_size[1]),
                                          grid_size=self.grid_size,
                                          inference=inference)

        condition = superpixel_ret['condition']
        condition = condition.squeeze(-1)
        condition[condition < 0] = 0
        feature_ret = self.feature_lattice(grid_pos=pred_points,
                                           img_fea=bifilter_crop_image[:, :3, ...].permute(0, 2, 3, 1),
                                           base_triangle2point=base_triangle2point, grid_size=self.grid_size,
                                           inference=True)
        feature_condition = feature_ret['condition']
        feature_condition = feature_condition.squeeze(-1)
        feature_condition[feature_condition < 0] = 0

        # feature pooling
        layer1_f, grid_pixel_size = self.grid_pooling(layer1_f, feature_condition, return_size=True)

        # deep network propagation
        layer2_f = self.layer2(layer1_f)
        layer3_f = self.layer3(layer2_f)
        layer4_f = self.layer4(layer3_f)

        layer1_f_c = self.res1_concat(layer1_f)
        layer2_f_c = self.res2_concat(layer2_f)
        layer3_f_c = self.res3_concat(layer3_f)
        layer4_f_c = self.res4_concat(layer4_f)
        concat_features = torch.cat((layer1_f_c, layer2_f_c, layer3_f_c, layer4_f_c), dim=1)

        x = self.cls(concat_features)

        if not inference:
            aux = self.aux(layer3_f)
            h = (self.args.grid_size[0] + 2) // 8 * 8
            w = (self.args.grid_size[1] * 2 + 2) // 8 * 8
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            laplacian_loss = output['laplacian_energy']
            variance = superpixel_ret['variance']
            area_variance = superpixel_ret['area_variance']
            reconstruct_loss = superpixel_ret['reconstruct_loss']
            deform_loss = variance * 1.0 + \
                          area_variance * 0.003 + \
                          laplacian_loss * 0.003 + \
                          reconstruct_loss * 0.5
            return x, condition, pred_points, aux, deform_loss
        return x, condition, pred_points

    def grid_pooling(self, src, condition, return_size=False):
        '''
        src: (batch, feature_dim, H, W)
        condition: (batch, N_pixel)
        function get_grid_mean_feature: src=(batch, N_pixel, feature_dim)
        '''
        batch, feature_dim, H, W = src.shape
        src = src.reshape(batch, feature_dim, -1).permute(0, 2, 1).contiguous()
        grid_mean_feature, grid_pixel_size = get_grid_mean_feature(src, condition, self.matrix.grid_num)
        grid_mean_feature = grid_mean_feature.permute(0, 2, 1).reshape(batch, feature_dim, (self.grid_size[0]-1), -1).contiguous()
        if return_size:
            grid_pixel_size = grid_pixel_size.reshape(batch, 1, (self.grid_size[0]-1), -1)
            return grid_mean_feature, grid_pixel_size
        return grid_mean_feature

    def grid2image(self, grid_image, condition, H=512, W=1024):
        '''
        grid_image: (batch, feature_dim, grid_size[0]-1, (grid_size[1]-1)*2)
        '''
        batch, feature_dim, _, _ = grid_image.shape
        grid_image = grid_image.reshape(batch, feature_dim, -1).permute(0, 2, 1)
        image = torch.gather(input=grid_image, dim=1,
                             index=condition.long().unsqueeze(-1).expand(-1, -1, feature_dim)).float()
        image = image.permute(0, 2, 1).reshape(batch, feature_dim, H, W)
        return image
