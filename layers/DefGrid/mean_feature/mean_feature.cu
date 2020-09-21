#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <THC/THC.h>
#include <vector>
#include <torch/torch.h>
#include <torch/extension.h>

#define MAX_FEATURE_CHANNEL_NUM 642


template <typename scalar_t>
__global__ void MeanFeatureForwardKernel(
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> mean_feature,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> grid_size,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feature_map,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> condition,
    const int batch_size,
    const int pixel_num,
    const int feature_channel_num,
    const int grid_num
)
{
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_idx = present_thread % grid_num;
    int batch_idx = present_thread / grid_num;
    if (grid_idx >= grid_num || batch_idx >= batch_size) return;
        
    int tmp_grid_size = 0;
    scalar_t tmp_mean_feature[MAX_FEATURE_CHANNEL_NUM] = {0};

    for (int pixel_idx = 0; pixel_idx < pixel_num; pixel_idx++){
        if (condition[batch_idx][pixel_idx] == grid_idx){
            tmp_grid_size++;
            for (int feature_channel_idx = 0; feature_channel_idx < feature_channel_num; feature_channel_idx++){

                tmp_mean_feature[feature_channel_idx] += feature_map[batch_idx][pixel_idx][feature_channel_idx];

            }
        }
    }
    if (tmp_grid_size == 0){
        return;
    }
    else{
        for (int feature_channel_idx = 0; feature_channel_idx < feature_channel_num; feature_channel_idx++){

            scalar_t tmp_feature = tmp_mean_feature[feature_channel_idx] / tmp_grid_size;
            mean_feature[batch_idx][grid_idx][feature_channel_idx] = tmp_feature;
        }
        grid_size[batch_idx][grid_idx] = tmp_grid_size;
    }

}

void MeanFeatureForwardKernelLauncher(
    at::Tensor mean_feature,
    at::Tensor grid_size,
    const at::Tensor feature_map,
    const at::Tensor condition,
    const int batch_size,
    const int pixel_num,
    const int feature_channel_num,
    const int grid_num
)
{

    const int thread_num = 1024;
    const int total_thread_num = batch_size * grid_num;
    const int block_num = total_thread_num / thread_num + 1;
    
    const dim3 threads(thread_num, 1, 1);
    const dim3 blocks(block_num, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(feature_map.type(), "mean_feature_forward_cuda", ([&] {
        MeanFeatureForwardKernel<scalar_t><<<blocks, threads>>>(
            mean_feature.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            grid_size.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feature_map.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            condition.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            batch_size,
            pixel_num,
            feature_channel_num,
            grid_num
        );
    }));

}