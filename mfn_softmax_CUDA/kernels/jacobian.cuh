#ifndef MFN_JACOBIAN_H_INCLUDED
#define MFN_JACOBIAN_H_INCLUDED

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/core/TensorBase.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

//add following line otherwise TensorInfo can not be identified
using namespace at::cuda::detail;

namespace JACOBIAN{

    void mfn_jacobian(
        TensorInfo<const float, int>(output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<float, int>(jacobian)
    );

    void mfn_positional_jacobian(
        TensorInfo<const float, int>(output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<const float, int>(positional_weights),
        TensorInfo<float, int>(jacobian)
    );
}

#endif
