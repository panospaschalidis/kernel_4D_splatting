#include <torch/extension.h>
#include "mfn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mfn_forward", &mfn_forward);
  m.def("mfn_backward", &mfn_backward);
  m.def("mfn_jacobian", &mfn_jacobian);
  m.def("mfn_positional_jacobian", &mfn_positional_jacobian);
}
