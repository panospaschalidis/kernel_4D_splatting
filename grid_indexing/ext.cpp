#include <torch/extension.h>
#include "indexing.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexing", &indexing);
}
