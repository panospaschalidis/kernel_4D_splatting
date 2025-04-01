#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/core/TensorBase.h>
#include "mfn.h"
#include "kernels/forward.cuh"
#include "kernels/backward.cuh"
#include "kernels/jacobian.cuh"
#include "kernels/config.h"


using namespace at::cuda::detail;

torch::Tensor mfn_forward(
                        const torch::Tensor& input,
                        const torch::Tensor& indices,
                        const torch::Tensor& W1,
                        const torch::Tensor& b1, 
                        const torch::Tensor& W2,
                        const torch::Tensor& b2,
                        const torch::Tensor& Wout,
                        const torch::Tensor& bout
                        ){
    auto output = torch::zeros_like(indices);
    FORWARD::mfn_forward(
        getTensorInfo<const float, int>(input.contiguous()),
        getTensorInfo<const float, int>(indices.contiguous()),
        getTensorInfo<const float, int>(W1.contiguous()),
        getTensorInfo<const float, int>(b1.contiguous()),
        getTensorInfo<const float, int>(W2.contiguous()),
        getTensorInfo<const float, int>(b2.contiguous()),
        getTensorInfo<const float, int>(Wout.contiguous()),
        getTensorInfo<const float, int>(bout.contiguous()),
        getTensorInfo<float, int>(output.contiguous())
    );
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
mfn_backward(
            const torch::Tensor& grad_output,
            const torch::Tensor& output,
            const torch::Tensor& input,
            const torch::Tensor& indices,
            const torch::Tensor& W1,
            const torch::Tensor& b1, 
            const torch::Tensor& W2,
            const torch::Tensor& b2,
            const torch::Tensor& Wout,
            const torch::Tensor& bout
            ){
    auto grad_W1 = torch::zeros_like(W1);
    auto grad_b1 = torch::zeros_like(b1);
    auto grad_W2 = torch::zeros_like(W2);
    auto grad_b2 = torch::zeros_like(b2);
    auto grad_Wout = torch::zeros_like(Wout);
    auto grad_bout = torch::zeros_like(bout);
    BACKWARD::mfn_backward(
        getTensorInfo<const float, int>(grad_output.contiguous()),
        getTensorInfo<const float, int>(output.contiguous()),
        getTensorInfo<const float, int>(input.contiguous()),
        getTensorInfo<const float, int>(indices.contiguous()),
        getTensorInfo<const float, int>(W1.contiguous()),
        getTensorInfo<const float, int>(b1.contiguous()),
        getTensorInfo<const float, int>(W2.contiguous()),
        getTensorInfo<const float, int>(b2.contiguous()),
        getTensorInfo<const float, int>(Wout.contiguous()),
        getTensorInfo<const float, int>(bout.contiguous()),
        getTensorInfo<float, int>(grad_W1.contiguous()),
        getTensorInfo<float, int>(grad_b1.contiguous()),
        getTensorInfo<float, int>(grad_W2.contiguous()),
        getTensorInfo<float, int>(grad_b2.contiguous()),
        getTensorInfo<float, int>(grad_Wout.contiguous()),
        getTensorInfo<float, int>(grad_bout.contiguous())
    );
    return std::make_tuple(grad_W1, grad_b1, grad_W2, grad_b2, grad_Wout, grad_bout);
}
//
torch::Tensor mfn_jacobian(
                        const torch::Tensor& output,
                        const torch::Tensor& input,
                        const torch::Tensor& indices,
                        const torch::Tensor& W1,
                        const torch::Tensor& b1, 
                        const torch::Tensor& W2,
                        const torch::Tensor& b2,
                        const torch::Tensor& Wout,
                        const torch::Tensor& bout
                        ){
        auto jacobian = torch::zeros(
            {output.size(0), output.size(1), input.size(1)},
            input.options());
        JACOBIAN::mfn_jacobian(
            getTensorInfo<const float, int>(output.contiguous()),
            getTensorInfo<const float, int>(input.contiguous()),
            getTensorInfo<const float, int>(indices.contiguous()),
            getTensorInfo<const float, int>(W1.contiguous()),
            getTensorInfo<const float, int>(b1.contiguous()),
            getTensorInfo<const float, int>(W2.contiguous()),
            getTensorInfo<const float, int>(b2.contiguous()),
            getTensorInfo<const float, int>(Wout.contiguous()),
            getTensorInfo<const float, int>(bout.contiguous()),
            getTensorInfo<float, int>(jacobian.contiguous())
        );
        return jacobian;
}

torch::Tensor mfn_positional_jacobian(
                        const torch::Tensor& output,
                        const torch::Tensor& input,
                        const torch::Tensor& indices,
                        const torch::Tensor& W1,
                        const torch::Tensor& b1, 
                        const torch::Tensor& W2,
                        const torch::Tensor& b2,
                        const torch::Tensor& Wout,
                        const torch::Tensor& bout,
                        const torch::Tensor& positional_weights
                        ){
    auto jacobian = torch::zeros(
        {output.size(0), output.size(1), positional_weights.size(1)},
        input.options());
    JACOBIAN::mfn_positional_jacobian(
        getTensorInfo<const float, int>(output.contiguous()),
        getTensorInfo<const float, int>(input.contiguous()),
        getTensorInfo<const float, int>(indices.contiguous()),
        getTensorInfo<const float, int>(W1.contiguous()),
        getTensorInfo<const float, int>(b1.contiguous()),
        getTensorInfo<const float, int>(W2.contiguous()),
        getTensorInfo<const float, int>(b2.contiguous()),
        getTensorInfo<const float, int>(Wout.contiguous()),
        getTensorInfo<const float, int>(bout.contiguous()),
        getTensorInfo<const float, int>(positional_weights.contiguous()),
        getTensorInfo<float, int>(jacobian.contiguous())
    );

    return jacobian;

}
