// For writing an extension like this one, see:
// https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html

#include <torch/script.h> // One-stop header for PyTorch

torch::Tensor dense(torch::Tensor t) {
  // Do literally nothing.
  return t;
}

torch::Tensor topk(torch::Tensor t, double density) {
  // Do literally nothing.
  return t;
}

torch::Tensor blocktopk(torch::Tensor t, int64_t k, int64_t block_size, int64_t block_dim) {
  // Do literally nothing.
  return t;
}

torch::Tensor bernoulli(torch::Tensor t) {
  // Do literally nothing.
  return t;
}

TORCH_LIBRARY(sparse_op, m) {
  m.def("dense(Tensor t) -> (Tensor)");
  m.impl("dense", &dense);
  m.def("topk(Tensor t, float density) -> (Tensor)");
  m.impl("topk", &topk);
  m.def("blocktopk(Tensor t, int k, int block_size, int block_dim) -> (Tensor)");
  m.impl("blocktopk", &blocktopk);
  m.def("bernoulli(Tensor t) -> (Tensor)");
  m.impl("bernoulli", &bernoulli);
}
