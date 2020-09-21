#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <THC/THC.h>
#include <vector>
#include <torch/torch.h>
#include <torch/extension.h>

#define eps 1e-10
#define SCALE 1.0
#define MAX_DIS  9999999999.0

template<typename scalar_t>
__host__ __device__ scalar_t line_variance_topk_cuda_abs(scalar_t a){
	if (a > 0.0){
		return a;
	}
	else{
		return -a;

	}
}

template<typename scalar_t>
__host__ __device__ scalar_t line_variance_topk_cuda_sign(scalar_t a){
	if (a > 0.0){
		return 1;
	}
	else if (a == 0.0){
		return 0;
	}
	else{
		return -1;
	}
}

template<typename scalar_t>
__host__ __device__ scalar_t line_variance_topk_cuda_square(scalar_t a){
	return a * a;
}


template<typename scalar_t>
__host__ __device__ scalar_t line_variance_topk_cuda_min_dis(scalar_t a, scalar_t b, scalar_t c){
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
__host__ __device__ scalar_t line_variance_topk_cuda_min_dis_idx(scalar_t a, scalar_t b, scalar_t c){
	scalar_t min_d = a;
	int min_idx = 0;
	if (b < min_d){
		min_d = b;
		min_idx = 1;
	}
	if (c < min_d){
		min_d = c;
		min_idx = 2;
	}
	return min_idx;
}

template<typename scalar_t>
__host__ __device__ scalar_t line_variance_topk_cuda_divide_non_zero(scalar_t a){
	if (a == 0){
		return eps;
	}
	if (a < 0){
		return a - eps;
	}
	if (a > 0){
		return a + eps;
	}
	return eps;
}

template <typename scalar_t>
__host__ __device__ scalar_t distance_line(scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2, scalar_t x, scalar_t y){
	
	scalar_t dx1x2 = -x1 + x2;
	scalar_t dy1y2 = -y1 + y2;
	scalar_t dx1x = x - x1;
	scalar_t dy1y = y - y1;
	
	scalar_t c1 = - x * x1 + x * x2 + x1 * x1 - x1 * x2 - y * y1 + y * y2 + y1 * y1 - y1 * y2;
	scalar_t c2 = x1 * x1 - 2 * x1 * x2 + x2 * x2 + y1 * y1  - 2 * y1 * y2 + y2 * y2;
	
	scalar_t d1 = -dx1x + dx1x2 * c1 / line_variance_topk_cuda_divide_non_zero(c2);
	scalar_t d2 = -dy1y + dy1y2 * c1 / line_variance_topk_cuda_divide_non_zero(c2);
	
	scalar_t dis = 	line_variance_topk_cuda_abs(d1)	+ line_variance_topk_cuda_abs(d2);

	return dis;
}
template <typename scalar_t>
__host__ __device__ scalar_t distance_point(scalar_t x1, scalar_t y1, scalar_t x, scalar_t y){
	return line_variance_topk_cuda_abs(x - x1) + line_variance_topk_cuda_abs(y - y1);
}

template <typename scalar_t>
__host__ __device__ void cal_line_gradient(scalar_t* grad, scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2, scalar_t x, scalar_t y){
	
	scalar_t dx1x2 = -x1 + x2;
	scalar_t dy1y2 = -y1 + y2;
	scalar_t dx1x = x - x1;
	scalar_t dy1y = y - y1;
	
	scalar_t c1 = - x * x1 + x * x2 + x1 * x1 - x1 * x2 - y * y1 + y * y2 + y1 * y1 - y1 * y2;
	scalar_t c2 = x1 * x1 - 2 * x1 * x2 + x2 * x2 + y1 * y1  - 2 * y1 * y2 + y2 * y2;
	scalar_t c12 = c1 / line_variance_topk_cuda_divide_non_zero(c2 * c2);
	
	scalar_t cx = - dx1x - dx1x2;
	scalar_t cy = - dy1y - dy1y2;
	
	scalar_t d1 = - dx1x + dx1x2 * c1 / line_variance_topk_cuda_divide_non_zero(c2);
	scalar_t d2 = - dy1y + dy1y2 * c1 / line_variance_topk_cuda_divide_non_zero(c2);
	
	
	//scalar_t dis = line_variance_topk_cuda_abs(d1) + line_variance_topk_cuda_abs(d2);
	
	scalar_t dif_x1 = (2 * dx1x2 * dy1y2 * c12 + dy1y2 * cx / line_variance_topk_cuda_divide_non_zero(c2)) * line_variance_topk_cuda_sign(d2) + (2 * dx1x2 * dx1x2 * c12 + dx1x2 * cx / line_variance_topk_cuda_divide_non_zero(c2) + 1 - c1 / line_variance_topk_cuda_divide_non_zero(c2)) * line_variance_topk_cuda_sign(d1);
	scalar_t dif_y1 = (2 * dx1x2 * dy1y2 * c12 + dx1x2 * cy / line_variance_topk_cuda_divide_non_zero(c2)) * line_variance_topk_cuda_sign(d1) + (2 * dy1y2 * dy1y2 * c12 + dy1y2 * cy / line_variance_topk_cuda_divide_non_zero(c2) + 1 - c1 / line_variance_topk_cuda_divide_non_zero(c2)) * line_variance_topk_cuda_sign(d2);
	scalar_t dif_x2 = (dx1x * dy1y2 / line_variance_topk_cuda_divide_non_zero(c2) - 2 * dx1x2 * dy1y2 * c12) * line_variance_topk_cuda_sign(d2) + (dx1x * dx1x2 / line_variance_topk_cuda_divide_non_zero(c2) - 2 * dx1x2 * dx1x2 * c12 + c1 / line_variance_topk_cuda_divide_non_zero(c2)) * line_variance_topk_cuda_sign(d1);
	scalar_t dif_y2 = (dx1x2 * dy1y / line_variance_topk_cuda_divide_non_zero(c2) - 2 * dx1x2 * dy1y2 * c12) * line_variance_topk_cuda_sign(d1) + (dy1y * dy1y2 / line_variance_topk_cuda_divide_non_zero(c2) - 2 * dy1y2 * dy1y2 * c12 + c1 / line_variance_topk_cuda_divide_non_zero(c2)) * line_variance_topk_cuda_sign(d2);

	grad[0] = dif_x1;
	grad[1] = dif_y1;
	grad[2] = dif_x2;
	grad[3] = dif_y2;
}

template <typename scalar_t>
__host__ __device__ void distance(scalar_t* ret, scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2, scalar_t x3, scalar_t y3, scalar_t x, scalar_t y)
{
	//https://en.wikipedia.org/wiki/Barycentric_coordinate_system
	
	scalar_t x1_x2 = x1 - x2;
	scalar_t y1_y2 = y1 - y2;
	scalar_t x1_x3 = x1 - x3;
	scalar_t y1_y3 = y1 - y3;
	scalar_t x2_x3 = x2 - x3;
	scalar_t y2_y3 = y2 - y3;
	
	scalar_t x_x1 = x - x1;
	scalar_t y_y1 = y - y1;
	scalar_t x_x2 = x - x2;
	scalar_t y_y2 = y - y2;
	scalar_t x_x3 = x - x3;
	scalar_t y_y3 = y - y3;

	scalar_t k1 = y2_y3 * x_x3 - x2_x3 * y_y3;
	scalar_t k2 = x1_x3 * y_y3 - y1_y3 * x_x3;
	scalar_t k3 = y2_y3 * x1_x3 - x2_x3 * y1_y3;

	if(k3 == 0){ // not a legal triangle
		ret[0] = -2;
		return;
	}
	if(k3 > 0){ // clock-wise triangle
		ret[0] = -1;
		return;
	}

	scalar_t l1 = k1 / k3;
	scalar_t l2 = k2 / k3;
	scalar_t l3 = 1 - l1 - l2;

	scalar_t dis12 = distance_line(x1, y1, x2, y2, x, y);
	scalar_t dis23 = distance_line(x2, y2, x3, y3, x, y);
	scalar_t dis13 = distance_line(x1, y1, x3, y3, x, y);

	if (l1 >= 0 && l2 >= 0 && l3 >= 0){ // lie inside or on the boundary
		
		ret[0] = 0;
		scalar_t min_dis_line = line_variance_topk_cuda_min_dis(dis12, dis23, dis13);
		scalar_t min_dis_line_idx = line_variance_topk_cuda_min_dis_idx(dis12, dis23, dis13);
		ret[1] = min_dis_line;
		ret[2] = min_dis_line_idx;
		return;
	}
	// whether point can calculate distance to certain line
	bool within12 = ((y1_y2 * y_y1 + x_x1 * x1_x2) * (y1_y2 * y_y2 + x_x2 * x1_x2)) <= 0;
	bool within23 = ((y2_y3 * y_y3 + x_x3 * x2_x3) * (y2_y3 * y_y2 + x_x2 * x2_x3)) <= 0;	
	bool within13 = ((y1_y3 * y_y1 + x_x1 * x1_x3) * (y1_y3 * y_y3 + x_x3 * x1_x3)) <= 0;

	dis12 = within12 ? dis12 : MAX_DIS;
	dis23 = within23 ? dis23 : MAX_DIS;
	dis13 = within13 ? dis13 : MAX_DIS;

	scalar_t min_dis_line = line_variance_topk_cuda_min_dis(dis12, dis23, dis13);
	scalar_t min_dis_line_idx = line_variance_topk_cuda_min_dis_idx(dis12, dis23, dis13);

	scalar_t d1 = distance_point(x1, y1, x, y);
	scalar_t d2 = distance_point(x2, y2, x, y);  
	scalar_t d3 = distance_point(x3, y3, x, y);
	
	scalar_t min_dis_point = line_variance_topk_cuda_min_dis(d1, d2, d3);
	scalar_t min_dis_point_idx = line_variance_topk_cuda_min_dis_idx(d1, d2, d3);

	if (min_dis_line < min_dis_point){ //distance to line
		ret[0] = 1;
		ret[1] = min_dis_line;
		ret[2] = min_dis_line_idx;
	}
	else{ //distance to point
		ret[0] = 2;
		ret[1] = min_dis_point;
		ret[2] = min_dis_point_idx;
	}

}
template<typename scalar_t>
__global__ void line_variance_topk_cuda_backword_kernel_batch(
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dldvariance_bxn,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> img_fea_bxnxd,
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> grid_fea_bxkxd,
		const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> grid_bxkx3x2,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> img_pos_bxnx2,
		const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dldreconstruct_bxnxd,
		torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> topk_grid_bxnxk,
		torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> buffer_bxnxk,
		scalar_t* __restrict__ dldgrid_bxkx3x2,
		int bnum, int n_pixel, int n_grid, int d_fea, float sigma, int topk){

	// bidx * height + heiidx
	int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

	int pixel_idx = presentthread % n_pixel;
	int bidx = (presentthread - pixel_idx) / n_pixel;

	if (bidx >= bnum || pixel_idx >= n_pixel)
		return;

	scalar_t pixel_x = img_pos_bxnx2[bidx][pixel_idx][0];
	scalar_t pixel_y = img_pos_bxnx2[bidx][pixel_idx][1];
	scalar_t x0 = pixel_x * SCALE;
	scalar_t y0 = pixel_y * SCALE;
	scalar_t x1, y1, x2, y2; // tmp variable for calculating the gradients
	scalar_t min_distance = 0.0;
	scalar_t sum_exp = 0.0;
	int min_distance_idx = 0;
	int idx_one = 0;
	int idx_two = 0;
	scalar_t find_sign = 0.0;
	scalar_t max_dist = -MAX_DIS;
    scalar_t ax, ay, bx, by, cx, cy;

	scalar_t ret[3] = {0};
	scalar_t grad[4] = {0};
	scalar_t condition;
	int grididx = 0;
	for (int k = 0; k < topk; k++){

        grididx = __float2int_rn(topk_grid_bxnxk[bidx][pixel_idx][k]);
		ax = grid_bxkx3x2[bidx][grididx][0][0] * SCALE;
		ay = grid_bxkx3x2[bidx][grididx][0][1] * SCALE;
		bx = grid_bxkx3x2[bidx][grididx][1][0] * SCALE;
		by = grid_bxkx3x2[bidx][grididx][1][1] * SCALE;
		cx = grid_bxkx3x2[bidx][grididx][2][0] * SCALE;
		cy = grid_bxkx3x2[bidx][grididx][2][1] * SCALE;

		distance(ret, ax, ay, bx, by, cx, cy, x0, y0);
		condition = ret[0];
		min_distance = ret[1];

		if (condition < 0) {
			min_distance = - MAX_DIS;
		}
		else if (condition == 0 && find_sign ==0){
			min_distance = min_distance / sigma;
			find_sign == 1;
		}
		else{
			min_distance = - min_distance / sigma;
		}
		max_dist = max_dist > min_distance ? max_dist : min_distance;
		buffer_bxnxk[bidx][pixel_idx][k] = min_distance;
	}

	for (int k = 0; k < topk; k++){
	    buffer_bxnxk[bidx][pixel_idx][k] = expf(buffer_bxnxk[bidx][pixel_idx][k] - max_dist);
		sum_exp += buffer_bxnxk[bidx][pixel_idx][k];
	}

	scalar_t sum_gradient = 0.0;
	scalar_t pixel_f = 0.0;
	scalar_t grid_f = 0.0;
	scalar_t diff = 0.0;
	scalar_t difference = 0.0;
	scalar_t grid_f_sum = 0.0;

	for (int k = 0; k < topk; k ++){
	    buffer_bxnxk[bidx][pixel_idx][k] = buffer_bxnxk[bidx][pixel_idx][k] / line_variance_topk_cuda_divide_non_zero(sum_exp);
	    grididx = __float2int_rn(topk_grid_bxnxk[bidx][pixel_idx][k]);
	    difference = 0.0;
	    grid_f_sum = 0.0;
		for (int d = 0; d < d_fea; d++){
			grid_f = grid_fea_bxkxd[bidx][grididx][d];
			pixel_f = img_fea_bxnxd[bidx][pixel_idx][d];
			diff = line_variance_topk_cuda_square(grid_f - pixel_f);
			difference = difference + diff;
			grid_f_sum += (dldreconstruct_bxnxd[bidx][pixel_idx][d] * grid_f);
		}
	    sum_gradient += (buffer_bxnxk[bidx][pixel_idx][k] * (dldvariance_bxn[bidx][pixel_idx] * difference + \
	                                                            grid_f_sum));
	}

    find_sign = 0.0;
    scalar_t dl_dmindist_element = 0.0;
	for (int k = 0; k < topk; k++){
	    grididx = __float2int_rn(topk_grid_bxnxk[bidx][pixel_idx][k]);
        scalar_t difference = 0.0;
        scalar_t grid_f_sum = 0.0;
		for (int d = 0; d < d_fea; d++){
			grid_f = grid_fea_bxkxd[bidx][grididx][d];
			pixel_f = img_fea_bxnxd[bidx][pixel_idx][d];
			diff = line_variance_topk_cuda_square(grid_f - pixel_f);
			difference = difference + diff;
			grid_f_sum += (dldreconstruct_bxnxd[bidx][pixel_idx][d] * grid_f);
		}
        dl_dmindist_element = buffer_bxnxk[bidx][pixel_idx][k] * (dldvariance_bxn[bidx][pixel_idx] * difference + grid_f_sum) - \
                                sum_gradient * buffer_bxnxk[bidx][pixel_idx][k];

		ax = grid_bxkx3x2[bidx][grididx][0][0] * SCALE;
		ay = grid_bxkx3x2[bidx][grididx][0][1] * SCALE;
		bx = grid_bxkx3x2[bidx][grididx][1][0] * SCALE;
		by = grid_bxkx3x2[bidx][grididx][1][1] * SCALE;
		cx = grid_bxkx3x2[bidx][grididx][2][0] * SCALE;
		cy = grid_bxkx3x2[bidx][grididx][2][1] * SCALE;

		distance(ret, ax, ay, bx, by, cx, cy, x0, y0);
		condition = ret[0];
		min_distance = ret[1];
		min_distance_idx = ret[2];

		int mem_gradient_idx = bidx * n_grid * 3 * 2  + grididx * 3 * 2;
		float in_out_sign;

		if (condition < 0){
			continue;
		}
		if (condition == 0 || condition == 1){
			in_out_sign = 1 - condition * 2;
			idx_one = min_distance_idx;
			idx_two = (min_distance_idx + 1 ) % 3;
			x1 = grid_bxkx3x2[bidx][grididx][idx_one][0];
			y1 = grid_bxkx3x2[bidx][grididx][idx_one][1];
			x2 = grid_bxkx3x2[bidx][grididx][idx_two][0];
			y2 = grid_bxkx3x2[bidx][grididx][idx_two][1];
			cal_line_gradient(grad, x1, y1, x2, y2, x0, y0);
			atomicAdd((float *)(dldgrid_bxkx3x2 + (mem_gradient_idx + idx_one * 2)), float(dl_dmindist_element * grad[0] / sigma * in_out_sign));
			atomicAdd((float *)(dldgrid_bxkx3x2 + (mem_gradient_idx + idx_one * 2 + 1)), float(dl_dmindist_element * grad[1] / sigma * in_out_sign));
			atomicAdd((float *)(dldgrid_bxkx3x2 + (mem_gradient_idx + idx_two * 2)), float(dl_dmindist_element * grad[2] / sigma * in_out_sign));
			atomicAdd((float *)(dldgrid_bxkx3x2 + (mem_gradient_idx + idx_two * 2 + 1)), float(dl_dmindist_element * grad[3] / sigma * in_out_sign));
		}
		else{
			in_out_sign = -1;
			x1 = grid_bxkx3x2[bidx][grididx][min_distance_idx][0];
			y1 = grid_bxkx3x2[bidx][grididx][min_distance_idx][0];
			float signx, signy;
			if (x1 > x0){
				signx = 1;
			}
			else{
				signx = -1;
			}
			if (y1 > y0){
				signy = 1;
			}
			else{
				signy = -1;
			}
			atomicAdd((float *)(dldgrid_bxkx3x2 + (mem_gradient_idx + min_distance_idx * 2)), float(signx * dl_dmindist_element / sigma * in_out_sign));
			atomicAdd((float *)(dldgrid_bxkx3x2 + (mem_gradient_idx + min_distance_idx * 2 + 1)), float(signy * dl_dmindist_element / sigma * in_out_sign));
		}
		
	}
}

void line_variance_topk_cuda_backward_batch(at::Tensor dldvariance_bxn, at::Tensor img_fea_bxnxd, at::Tensor grid_fea_bxkxd, at::Tensor grid_bxkx3x2, at::Tensor img_pos_bxnx2,
                        float sigma, at::Tensor dldreconstruct_bxnxd, at::Tensor topk_grid_bxnxk, at::Tensor buffer_bxnxk, at::Tensor dldgrid_bxkx3x2) {

	int bnum = grid_bxkx3x2.size(0);
	int n_grid = grid_bxkx3x2.size(1);
	int n_pixel = img_pos_bxnx2.size(1);
	int d_fea = img_fea_bxnxd.size(2);
	int topk = topk_grid_bxnxk.size(2);
	// for fxbxhxw image size
	const int threadnum = 512;
	const int totalthread = bnum * n_pixel;
	const int blocknum = totalthread / threadnum + 1;

	const dim3 threads(threadnum, 1, 1);
	const dim3 blocks(blocknum, 1, 1);

	// we exchange block and thread!
	AT_DISPATCH_FLOATING_TYPES(grid_bxkx3x2.type(), "line_variance_topk_cuda_backward_batch",
			([&] {
				line_variance_topk_cuda_backword_kernel_batch<scalar_t><<<blocks, threads>>>(
				        dldvariance_bxn.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
						img_fea_bxnxd.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                        grid_fea_bxkxd.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                        grid_bxkx3x2.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
                        img_pos_bxnx2.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                        dldreconstruct_bxnxd.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                        topk_grid_bxnxk.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                        buffer_bxnxk.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
                        dldgrid_bxkx3x2.data<scalar_t>(),
                        bnum, n_pixel, n_grid, d_fea, sigma, topk);
			}));

	return;
}

