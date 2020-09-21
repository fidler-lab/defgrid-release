#include <torch/torch.h>
#include <THC/THC.h>
#include <vector>
#include <stdio.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DIM4(x, b, h, w, d) AT_ASSERTM((x.size(0) == b) && (x.size(1) == h) && (x.size(2) == w) && (x.size(3) == d), #x "must be the same size")
#define CHECK_DIM3(x, b, h, w) AT_ASSERTM((x.size(0) == b) && (x.size(1) == h) && (x.size(2) == w), #x "must be the same size")
#define CHECK_DIM2(x, b, h) AT_ASSERTM((x.size(0) == b) && (x.size(1) == h), #x "must be the same size")

void MeanFeatureForwardKernelLauncher(
    at::Tensor mean_feature,
    at::Tensor grid_size,
    const at::Tensor feature_map,
    const at::Tensor condition,
    const int batch_size,
    const int pixel_num,
    const int feature_channel_num,
    const int grid_num
);

void mean_feature_forward_cuda(
    at::Tensor mean_feature, 
    at::Tensor grid_size, 
    const at::Tensor feature_map, 
    const at::Tensor condition
)
{

    CHECK_INPUT(mean_feature);
    CHECK_INPUT(grid_size);
    CHECK_INPUT(feature_map);
    CHECK_INPUT(condition);

    int batch_size = feature_map.size(0);
    int pixel_num = feature_map.size(1);
    int feature_channel_num = feature_map.size(2);
    int grid_num = mean_feature.size(1);

    CHECK_DIM3(mean_feature, batch_size, grid_num, feature_channel_num);
    CHECK_DIM2(grid_size, batch_size, grid_num);
    CHECK_DIM3(feature_map, batch_size, pixel_num, feature_channel_num);
    CHECK_DIM2(condition, batch_size, pixel_num);

    MeanFeatureForwardKernelLauncher(mean_feature, grid_size, feature_map, condition, batch_size, pixel_num, feature_channel_num, grid_num);
    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward_cuda", &mean_feature_forward_cuda, "mean feature gather for grids (CUDA)");
}