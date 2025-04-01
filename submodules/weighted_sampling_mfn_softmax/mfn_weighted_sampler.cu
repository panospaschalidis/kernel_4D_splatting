#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/core/TensorBase.h>
#include "mfn_weighted_sampler.h"
#include "impl/forward.cuh"
#include "impl/backward.cuh"
#include "impl/config.h"



using namespace at::cuda::detail;

torch::Tensor mfn_sampler_forward(
    const torch::Tensor& input, 
    const torch::Tensor& grid,
    const torch::Tensor& weights)
{

    int N = input.contiguous().size(0);
    int C = input.contiguous().size(1);
    int inp_H = input.contiguous().size(2);
    int inp_W = input.contiguous().size(3);
    int out_H = grid.contiguous().size(1);
    int out_W = grid.contiguous().size(2);
    auto output = torch::empty({N, C, out_H, out_W}, input.options());
    FORWARD::sampler_CUDA <<<((N*out_H*out_W)+threads_pb-1)/threads_pb, threads_pb >>>(
        getTensorInfo<const float, int>(input.contiguous()),
        getTensorInfo<const float, int>(grid.contiguous()),
        getTensorInfo<const float, int>(weights.contiguous()),
        getTensorInfo<float, int>(output.contiguous())
    );
    return output; 
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
mfn_sampler_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input, 
    const torch::Tensor& grid,
    const torch::Tensor& weights,
    const torch::Tensor& weights_jacobian)
{
    
    auto grad_input = torch::zeros_like(input);
    auto grad_grid = torch::zeros_like(grid);
    auto grad_weights = torch::zeros_like(weights);
    //auto grad_grid = torch::ones_like(grad_output);
    auto grad_input_numel = input.numel();
    if (grid.numel() == 0 || input.numel() == 0) {
    grad_grid.zero_();
    return std::make_tuple(grad_input, grad_grid, grad_weights);
    }
    int N = input.contiguous().size(0);
    int C = input.contiguous().size(1);
    int inp_H = input.contiguous().size(2);
    int inp_W = input.contiguous().size(3);
    int out_H = grid.contiguous().size(1);
    int out_W = grid.contiguous().size(2);
    BACKWARD::sampler_CUDA <<<((N*out_H*out_W)+threads_pb-1)/threads_pb, threads_pb >>>(
        getTensorInfo<const float, int>(grad_output.contiguous()),
        getTensorInfo<const float, int>(input.contiguous()),
        getTensorInfo<const float, int>(grid.contiguous()),
        getTensorInfo<const float, int>(weights.contiguous()),
        getTensorInfo<const float, int>(weights_jacobian.contiguous()),
        getTensorInfo<float, int>(grad_input.contiguous()),
        getTensorInfo<float, int>(grad_grid.contiguous()),
        getTensorInfo<float, int>(grad_weights.contiguous()),
        grad_input_numel
    );
    return std::make_tuple(grad_input, grad_grid, grad_weights);
}

