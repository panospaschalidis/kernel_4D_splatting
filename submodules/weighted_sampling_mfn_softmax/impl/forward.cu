#include <iostream>
#include <torch/extension.h>
#include "forward.cuh"
#include "config.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;



__global__ void FORWARD::sampler_CUDA(
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(grid),
        TensorInfo<const float, int>(weights),
        TensorInfo<float, int>(output))
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
    int out_sN = output.strides[0];
    int out_sC = output.strides[1];
    int out_sH = output.strides[2];
    int out_sW = output.strides[3];
    auto thread = cg::this_grid().thread_rank();
    auto block = cg::this_thread_block();
    if (thread >= N*out_H*out_W)
        return;
    int thread_pos = (int)thread;
    float x = grid.data[2*thread_pos];
    float y = grid.data[2*thread_pos+1];
    
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
    //int inp_ptr = (static_cast<int>(thread/(ss[3]))) * ss[9];
    int inp_ptr = 0;
    for (int c = 0; c < C; ++c, out_ptr += out_sC, inp_ptr += inp_sC) {
        auto res = static_cast<float>(0);
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
            res += input.data[inp_ptr + iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
            res += input.data[inp_ptr + iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
            res += input.data[inp_ptr + iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
            res += input.data[inp_ptr + iy_se * inp_sH + ix_se * inp_sW] * se;
        }
        output.data[out_ptr + thread_pos] = res;
    }
}
