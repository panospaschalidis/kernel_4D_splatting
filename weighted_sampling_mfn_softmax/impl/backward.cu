#include <iostream>
#include <torch/extension.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include "backward.cuh"
#include "config.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

using at::native::fastAtomicAdd;

__global__ void BACKWARD::sampler_CUDA(
        TensorInfo<const float, int>(grad_output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(grid),
        TensorInfo<const float, int>(weights),
        TensorInfo<const float, int>(weights_jacobian),
        TensorInfo<float, int>(grad_input),
        TensorInfo<float, int>(grad_grid),
        TensorInfo<float, int>(grad_weights),
        int grad_input_numel)
{   
    int N = input.sizes[0];
    int C = input.sizes[1];
    int inp_H = input.sizes[2];
    int inp_W = input.sizes[3];
    int out_H = grid.sizes[1];
    int out_W = grid.sizes[2];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sH = input.strides[2];
    int inp_sW = input.strides[3];
    int grid_sN = grid.strides[0];
    int grid_sH = grid.strides[1];
    int grid_sW = grid.strides[2];
    int grid_sCoor = grid.strides[3];
    int out_sN = grad_output.strides[0];
    int out_sC = grad_output.strides[1];
    int out_sH = grad_output.strides[2];
    int out_sW = grad_output.strides[3];
    auto thread = cg::this_grid().thread_rank();
    auto block = cg::this_thread_block();
    if (thread >= N*out_H*out_W)
        return;
    int thread_pos = (int)thread;
    float x = grid.data[2*thread_pos];
    float y = grid.data[2*thread_pos+1];
    float gix_mult, giy_mult;
    float ix = compute_index(x, inp_W);
    float iy = compute_index(y, inp_H);

    int ix_nw = static_cast<int>(std::floor(ix));
    int iy_nw = static_cast<int>(std::floor(iy));

    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;

    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;

    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;


    // assign weights to directional variables
    float nw = weights.data[4*thread_pos];
    float ne = weights.data[4*thread_pos+1];
    float sw = weights.data[4*thread_pos+2];
    float se = weights.data[4*thread_pos+3];
    int out_ptr = 0;
    //int inp_ptr = (static_cast<int>(thread/(inp_W*inp_H))) * ss[8];
    int inp_ptr = 0;
    float gix = static_cast<float>(0);
    float giy = static_cast<float>(0);
    float wix = static_cast<float>(0);
    float wiy = static_cast<float>(0);
    float wiz = static_cast<float>(0);
    float wiw = static_cast<float>(0);
    for (int c = 0; c < C; ++c, out_ptr += out_sC, inp_ptr += inp_sC) {
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            // you can also use fastAtomicAdd like in pytorch cuda Grid_sampler.cuh
            fastAtomicAdd(
                grad_input.data,
                inp_ptr + iy_nw * inp_sH + ix_nw * inp_sW, 
                grad_input_numel,
                grad_output.data[out_ptr+thread_pos]*nw, 
                true
            );
            float nw_val = input.data[inp_ptr + iy_nw * inp_sH + ix_nw * inp_sW];
            wix += nw_val * grad_output.data[out_ptr+thread_pos];
            gix += nw_val * weights_jacobian.data[thread_pos*8] * grad_output.data[out_ptr+thread_pos];
            giy += nw_val * weights_jacobian.data[thread_pos*8 + 1] * grad_output.data[out_ptr+thread_pos];
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            //atomicAdd(&(grad_input.data[inp_ptr + iy_ne * inp_sH + ix_ne * inp_sW]), grad_output.data[out_ptr+thread_pos]*ne);
            fastAtomicAdd(
                grad_input.data,
                inp_ptr + iy_ne * inp_sH + ix_ne * inp_sW, 
                grad_input_numel,
                grad_output.data[out_ptr+thread_pos]*ne,
                true
            );
            float ne_val = input.data[inp_ptr + iy_ne * inp_sH + ix_ne * inp_sW];
            wiy += ne_val * grad_output.data[out_ptr+thread_pos];
            gix += ne_val * weights_jacobian.data[thread_pos*8 + 2] * grad_output.data[out_ptr+thread_pos];
            giy += ne_val * weights_jacobian.data[thread_pos*8 + 3] * grad_output.data[out_ptr+thread_pos];
            //printf("thread_pos:%d, c:%d for input:%d, %d, value:%f\n",thread_pos, c, iy_ne, ix_ne, ne*grad_output.data[out_ptr+thread_pos]);
            // if you use thread, instead of thread_pos or (int)thread values are falsely displayed
            // it seems that inside the thread index controlling condition thread<= ... we have direct casting
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            //atomicAdd(&(grad_input.data[inp_ptr + iy_sw * inp_sH + ix_sw * inp_sW]), grad_output.data[out_ptr+thread_pos]*sw);
            fastAtomicAdd(
                grad_input.data,
                inp_ptr + iy_sw * inp_sH + ix_sw * inp_sW, 
                grad_input_numel,
                grad_output.data[out_ptr+thread_pos]*sw,
                true
            );
            float sw_val = input.data[inp_ptr + iy_sw * inp_sH + ix_sw * inp_sW];
            wiz += sw_val * grad_output.data[out_ptr+thread_pos];
            gix += sw_val * weights_jacobian.data[thread_pos*8 + 4] * grad_output.data[out_ptr+thread_pos];
            giy += sw_val * weights_jacobian.data[thread_pos*8 + 5] * grad_output.data[out_ptr+thread_pos];
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            //atomicAdd(&(grad_input.data[inp_ptr + iy_se * inp_sH + ix_se * inp_sW]), grad_output.data[out_ptr+thread_pos]*se);
            fastAtomicAdd(
                grad_input.data,
                inp_ptr + iy_se * inp_sH + ix_se * inp_sW, 
                grad_input_numel,
                grad_output.data[out_ptr+thread_pos]*se,
                true
            );
            float se_val = input.data[inp_ptr + iy_se * inp_sH + ix_se * inp_sW];
            wiw += se_val * grad_output.data[out_ptr+thread_pos];
            gix += se_val * weights_jacobian.data[thread_pos*8 + 6] * grad_output.data[out_ptr+thread_pos];
            giy += se_val * weights_jacobian.data[thread_pos*8 + 7] * grad_output.data[out_ptr+thread_pos];
        }
    }
    grad_grid.data[2*thread_pos] =  gix;
    grad_grid.data[2*thread_pos + 1] = giy;
    grad_weights.data[4*thread_pos] = wix;
    grad_weights.data[4*thread_pos + 1] = wiy;
    grad_weights.data[4*thread_pos + 2] = wiz;
    grad_weights.data[4*thread_pos + 3] = wiw;
}
