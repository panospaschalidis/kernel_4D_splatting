#include <iostream>
#include <torch/extension.h>

torch::Tensor mfn_forward(
                        const torch::Tensor& input,
                        const torch::Tensor& indices,
                        const torch::Tensor& W1,
                        const torch::Tensor& b1, 
                        const torch::Tensor& W2,
                        const torch::Tensor& b2,
                        const torch::Tensor& Wout,
                        const torch::Tensor& bout
                        );

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
            );

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
                        );

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
                        );
