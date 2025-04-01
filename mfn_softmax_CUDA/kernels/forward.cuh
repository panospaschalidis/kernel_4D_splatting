#ifndef MFN_FORWARD_H_INCLUDED
#define MFN_FORWARD_H_INCLUDED

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/core/TensorBase.h>

using namespace at::cuda::detail;

namespace FORWARD{

    void mfn_forward(
        TensorInfo<const float, int>(input),
        TensorInfo<const float, int>(indices),
        TensorInfo<const float, int>(W1),
        TensorInfo<const float, int>(b1),
        TensorInfo<const float, int>(W2),
        TensorInfo<const float, int>(b2),
        TensorInfo<const float, int>(Wout),
        TensorInfo<const float, int>(bout),
        TensorInfo<float, int>(output)
    );

}

#endif
