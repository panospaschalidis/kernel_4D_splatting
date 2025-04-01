#pragma once
#include <vector>
#include <iostream>
#include <torch/extension.h>

torch::Tensor mfn_sampler_forward(
    const torch::Tensor& input, 
    const torch::Tensor& grid,
    const torch::Tensor& weights);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
mfn_sampler_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input, 
    const torch::Tensor& grid,
    const torch::Tensor& weights,
    const torch::Tensor& weights_jacobian);

