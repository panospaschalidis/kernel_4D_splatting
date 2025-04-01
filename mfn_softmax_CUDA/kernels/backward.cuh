#ifndef MFN_BACKWARD_H_INCLUDED
#define MFN_BACKWARD_H_INCLUDED

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/core/TensorBase.h>
//add lines i#include <cuda.h> to #include<..._parameters.h>
//to avoid error: #error "GLM requires CUDA 7.0 or higher"
//caused by #include <glm/glm.hpp>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

//add following line otherwise TensorInfo can not be identified
using namespace at::cuda::detail;

namespace BACKWARD{

   void mfn_backward(
        TensorInfo<const float, int>(grad_output),
        TensorInfo<const float, int>(output),
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<float, int>(grad_W1),
        TensorInfo<float, int>(grad_b1),
        TensorInfo<float, int>(grad_W2),
        TensorInfo<float, int>(grad_b2),
        TensorInfo<float, int>(grad_Wout),
        TensorInfo<float, int>(grad_bout)
    );

}

#endif
