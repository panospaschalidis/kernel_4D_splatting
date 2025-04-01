#ifndef WEIGHTED_MFN_H_FORWARD_INCLUDED
#define WEIGHTED_MFN_H_FORWARD_INCLUDED

#include <iostream>
#include <torch/extension.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/core/TensorBase.h>


using namespace at::cuda::detail;

namespace FORWARD{

    __global__ void sampler_CUDA(
            TensorInfo<const float, int>(input),
            TensorInfo<const float, int>(grid),
            TensorInfo<const float, int>(weights),
            TensorInfo<float, int>(output));
}
#endif
