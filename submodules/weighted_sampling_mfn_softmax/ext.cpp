#include <torch/extension.h>
#include "mfn_weighted_sampler.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mfn_sampler_forward", &mfn_sampler_forward);
  m.def("mfn_sampler_backward", &mfn_sampler_backward);
}
