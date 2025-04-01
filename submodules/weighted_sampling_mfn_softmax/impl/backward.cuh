#ifndef WEIGHTED_MFN_H_BACKWARD_INCLUDED
#define WEIGHTED_MFN_H_BACKWARD_INCLUDED

#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/core/TensorBase.h>
 

using namespace at::cuda::detail;

namespace BACKWARD{
    
    __global__ void sampler_CUDA(
            TensorInfo<const float, int>(grad_output),
            TensorInfo<const float, int>(input),
            TensorInfo<const float, int>(grid),
            TensorInfo<const float, int>(weights),
            TensorInfo<const float, int>(weights_jacobian),
            TensorInfo<float, int>(grad_input),
            TensorInfo<float, int>(grad_grid),
            TensorInfo<float, int>(grad_weights),
            int grad_input_numel);

}

#endif
