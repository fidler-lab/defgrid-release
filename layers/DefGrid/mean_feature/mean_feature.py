import torch
import sys
from torch.utils.cpp_extension import load
import os
base_path = os.getcwd()

sys.path.append(os.path.join(base_path, 'layers/DefGrid/mean_feature'))
mean_feature_func = load(name="mean_feature",
            sources = [os.path.join(base_path, "layers/DefGrid/mean_feature/mean_feature.cpp"),
                       os.path.join(base_path, "layers/DefGrid/mean_feature/mean_feature.cu")],
            verbose=True)

class MeanFeatureGather(torch.autograd.Function):

    #expect feature map have feature channel at the last dimension, the second is the position dimension
    @staticmethod
    def forward(ctx, feature_map, condition, grid_num):

        device = feature_map.device

        if feature_map.dim() != 3:
            raise ValueError('expect feature_map to have exactly 3 dimensions')
        if condition.dim() != 2:
            raise ValueError('expect condition to have exactly 2 dimensions')

        batch_size, _, feature_channel_num = feature_map.shape
        assert (feature_channel_num <= 642)  # the max local buffer in .cu
        mean_feature = torch.zeros(batch_size, grid_num, feature_channel_num, device=feature_map.device)
        grid_size = torch.zeros(batch_size, grid_num, device=feature_map.device)
        condition = condition.contiguous()

        if feature_map.shape[1] == 100000000:
            for i_batch in range(feature_map.shape[0]):
                tmp_feature = feature_map[i_batch].unsqueeze(0).contiguous()
                mean_feature_func.forward_cuda(mean_feature[i_batch].unsqueeze(0), grid_size[i_batch].unsqueeze(0),
                                tmp_feature.float(), condition[i_batch].unsqueeze(0).float())

        else:
            feature_map = feature_map.contiguous()
            mean_feature_func.forward_cuda(mean_feature, grid_size, feature_map.float(), condition.float())

        ctx.save_for_backward(grid_size, condition)
        return mean_feature, grid_size

    @staticmethod
    def backward(ctx, grad_mean_feature, grad_grid_size):
        # If the shape is too big, we iterate every batch for this

        grid_size, condition = ctx.saved_tensors
        invalid_mask = condition < 0
        filtered_condition = condition.clone().long()
        filtered_condition[invalid_mask] = 0
        if filtered_condition.shape[-1] == 100000000:
            feature_map_grad_list = []
            for i_batch in range(filtered_condition.shape[0]):
                local_condition = filtered_condition[i_batch].unsqueeze(0)
                feature_map_weights = torch.gather(input=grid_size[i_batch].unsqueeze(0), dim=1,
                                                   index=local_condition).detach()  # for each pixel, how manny pixels in total its corresponding grid has
                tmp_shape = local_condition.shape
                feature_channel_num = grad_mean_feature[i_batch].shape[-1]
                tmp_condition = local_condition.unsqueeze(-1).expand(tmp_shape[0], tmp_shape[1], feature_channel_num)

                feature_map_grad = torch.gather(input=grad_mean_feature[i_batch].unsqueeze(0), dim=1,
                                                index=tmp_condition).detach()  # for each pixel, the grad vector of one position of its corresponding grid
                feature_map_grad = feature_map_grad / feature_map_weights.reshape(tmp_shape[0], tmp_shape[1],
                                                                                  1)  # divided by the total number of pixels one grid has
                feature_map_grad[invalid_mask[i_batch].unsqueeze(0)] = 0

                feature_map_grad_list.append(feature_map_grad)
            feature_map_grad = torch.cat(feature_map_grad_list, dim=0)
        else:
            feature_map_weights = torch.gather(input=grid_size, dim=1, index=filtered_condition).detach() #for each pixel, how manny pixels in total its corresponding grid has
            tmp_shape = filtered_condition.shape
            feature_channel_num = grad_mean_feature.shape[-1]
            tmp_condition = filtered_condition.unsqueeze(-1).expand(tmp_shape[0], tmp_shape[1], feature_channel_num)

            feature_map_grad = torch.gather(input=grad_mean_feature, dim=1, index=tmp_condition).detach() #for each pixel, the grad vector of one position of its corresponding grid
            feature_map_grad = feature_map_grad / feature_map_weights.reshape(tmp_shape[0], tmp_shape[1], 1) #divided by the total number of pixels one grid has
            feature_map_grad[invalid_mask] = 0

        return feature_map_grad, None, None

get_grid_mean_feature = MeanFeatureGather.apply
