#include <THC/THC.h>
#include <torch/torch.h>
#include <vector>
#include<stdio.h>
extern THCState *state;
// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_DIM3(x, b, h, w, d) AT_ASSERTM((x.size(0) == b) && (x.size(1) == h) && (x.size(2) == w) && (x.size(3) == d), #x " must be same im size")
#define CHECK_DIM2(x, b, f, d) AT_ASSERTM((x.size(0) == b) && (x.size(1) == f) && (x.size(2) == d), #x " must be same point size")
#define CHECK_DIM4(x, b, h, w, d, k) AT_ASSERTM((x.size(0) == b) && (x.size(1) == h) && (x.size(2) == w) && (x.size(3) == d) && (x.size(4) == k), #x " must be same im size")

void check_condition_cuda_forward_batch(at::Tensor grid_bxkx4x2, at::Tensor img_pos_bxnx2,  at::Tensor condition_bxnx1, at::Tensor bbox_bxkx2x2);

void check_condition_forward_batch(at::Tensor grid_bxkx3x2, at::Tensor img_pos_bxnx2,  at::Tensor condition_bxnx1, at::Tensor bbox_bxkx2x2) {
	CHECK_INPUT(grid_bxkx3x2);
	CHECK_INPUT(img_pos_bxnx2);
	CHECK_INPUT(condition_bxnx1);
	CHECK_INPUT(bbox_bxkx2x2);
	
	int bnum = grid_bxkx3x2.size(0);
	int n_grid = grid_bxkx3x2.size(1);
	int n_pixel = img_pos_bxnx2.size(1);
	CHECK_DIM3(grid_bxkx3x2, bnum, n_grid, 3, 2);
	CHECK_DIM2(img_pos_bxnx2, bnum, n_pixel, 2);
	CHECK_DIM2(condition_bxnx1, bnum, n_pixel, 1);
	CHECK_DIM3(bbox_bxkx2x2, bnum, n_grid, 2, 2);

	check_condition_cuda_forward_batch(grid_bxkx3x2, img_pos_bxnx2, condition_bxnx1, bbox_bxkx2x2);

	return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &check_condition_forward_batch, "check_condition forward batch (CUDA)");
}

