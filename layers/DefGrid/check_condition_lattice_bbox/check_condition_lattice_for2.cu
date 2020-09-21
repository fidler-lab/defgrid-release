#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <THC/THC.h>
#include <vector>
#define eps 1e-10
#define SCALE 1.0


template<typename scalar_t>
__host__ __device__ scalar_t check_condition_cuda_abs(scalar_t a){
	if (a > 0.0){
		return a;
	}
	else{
		return -a;
	}
}

template<typename scalar_t>
__host__ __device__ scalar_t check_condition_cuda_cross_multiple(scalar_t a_x, scalar_t a_y, scalar_t b_x, scalar_t b_y){
	return a_x * b_y - a_y * b_x;
}


template<typename scalar_t>
__host__ __device__ scalar_t check_condition_cuda_divide_non_zero(scalar_t a){

	if (a == 0){
		return eps;
	}
	if (a < 0){
		return a - eps;
	}
	if (a > 0){
		return a + eps;
	}
}

template<typename scalar_t>
__host__ __device__ scalar_t check_condition_cuda_min_dis(scalar_t a, scalar_t b, scalar_t c){
	scalar_t min_d = a;
	if (b < min_d){
		min_d = b;
	}
	if (c < min_d){
		min_d = c;
	}
	return min_d;
}

template<typename scalar_t>
__host__ __device__ scalar_t check_condition_cuda_mid_distance(scalar_t* __restrict__ a, scalar_t* __restrict__ b, scalar_t* __restrict__ c){
	// calculate the mid distance of a to bc line
	scalar_t a_x = a[0];
	scalar_t a_y = a[1];

	scalar_t b_x = b[0];
	scalar_t b_y = b[1];
	scalar_t c_x = c[0];
	scalar_t c_y = c[1];

	scalar_t mid_x = (b_x + c_x) / 2;
	scalar_t mid_y = (b_y + c_y) / 2;

	scalar_t distance = check_condition_cuda_abs(a_x - mid_x) + check_condition_cuda_abs(a_y - mid_y);
	return distance;
}

template<typename scalar_t>
__global__ void check_condition_cuda_forward_kernel_batch(
		scalar_t* __restrict__ grid_bxkx3x2,
		scalar_t* __restrict__ img_pos_bxnx2,
		scalar_t* __restrict__ condition_bxnx1,
		scalar_t* __restrict__ bbox_bxkx2x2,
		int bnum, int n_pixel, int n_grid)
		{
	// bidx * height + heiidx
	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;
	int pixel_idx = presentthread % n_pixel;
	int bidx = (presentthread - pixel_idx) / n_pixel;

	if (bidx >= bnum || pixel_idx >= n_pixel) {
		return;
	}

	/////////////////////////////////////////////////////////////////
	// which pixel it belongs to
	scalar_t pixel_x = img_pos_bxnx2[bidx * n_pixel * 2 + pixel_idx * 2];
	scalar_t pixel_y = img_pos_bxnx2[bidx * n_pixel * 2 + pixel_idx * 2 + 1];
	scalar_t x0 = pixel_x * SCALE;
	scalar_t y0 = pixel_y * SCALE;


	for (int grididx = 0; grididx < n_grid; grididx++){

	    scalar_t bbox_ax = bbox_bxkx2x2[bidx * n_grid * 2 * 2 + grididx * 2 *2 + 0];
	    scalar_t bbox_ay = bbox_bxkx2x2[bidx * n_grid * 2 * 2 + grididx * 2 *2 + 1];
	    scalar_t bbox_bx = bbox_bxkx2x2[bidx * n_grid * 2 * 2 + grididx * 2 *2 + 2];
	    scalar_t bbox_by = bbox_bxkx2x2[bidx * n_grid * 2 * 2 + grididx * 2 *2 + 3];
	    if (pixel_x < bbox_ax || pixel_x > bbox_bx || pixel_y < bbox_ay || pixel_y > bbox_by){
	        continue;
	    }

		// Check condition of in grid or outside of grid.
		// Time SCALE for numerical stability.
		scalar_t ax = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2] * SCALE;
		scalar_t ay = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 1] * SCALE;
		scalar_t bx = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 2] * SCALE;
		scalar_t by = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 3] * SCALE;
		scalar_t cx = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 4] * SCALE;
		scalar_t cy = grid_bxkx3x2[bidx * n_grid * 3 * 2 + grididx * 3 * 2 + 5] * SCALE;
		
		scalar_t condition = 0.0;

		// replace with other variables
		scalar_t m = bx - ax;
		scalar_t p = by - ay;

		scalar_t n = cx - ax;
		scalar_t q = cy - ay;

		scalar_t s = x0 - ax;
		scalar_t t = y0 - ay;

		scalar_t k1 = s * q - n * t;
		scalar_t k2 = m * t - s * p;
		scalar_t k3 = m * q - n * p;

		// is this face visible?

		if (k3 >= 0) {
			condition = 0.0;
		}
		else{
			scalar_t w1 = k1 / check_condition_cuda_divide_non_zero(k3);
			scalar_t w2 = k2 / check_condition_cuda_divide_non_zero(k3);
			scalar_t w0 = 1 - w1 - w2;
			// not lie in the triangle
			if (w0 < 0 || w1 < 0 || w2 < 0) {
				condition = 0.0;
			}
			else{
			    condition = 1.0;
			}
		}
		if (condition == 1.0){
		    condition_bxnx1[bidx * n_pixel + pixel_idx] = float(grididx);
		    break;
		}
	}
}

void check_condition_cuda_forward_batch(at::Tensor grid_bxkx3x2, at::Tensor img_pos_bxnx2, at::Tensor condition_bxnx1, at::Tensor bbox_bxkx2x2){

	int bnum = grid_bxkx3x2.size(0);
	int n_grid = grid_bxkx3x2.size(1);
	int n_pixel = img_pos_bxnx2.size(1);

	// for fxbxhxw image size
	const int threadnum = 1024;
	const int totalthread = bnum * n_pixel;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

//    cudaStream_t stream = THCState_getCurrentStream(state);
	AT_DISPATCH_FLOATING_TYPES(grid_bxkx3x2.type(), "check_condition_cuda_forward_batch", ([&] {
		check_condition_cuda_forward_kernel_batch<scalar_t><<<blocks, threads>>>(
				grid_bxkx3x2.data<scalar_t>(),
				img_pos_bxnx2.data<scalar_t>(),
				condition_bxnx1.data<scalar_t>(),
				bbox_bxkx2x2.data<scalar_t>(),
				bnum, n_pixel, n_grid);
	}));

	return;
}

