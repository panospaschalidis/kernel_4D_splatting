#include <torch/extension.h>
#include <iostream>

torch::Tensor mfn_forward(torch::Tensor z, int64_t grain_size);

