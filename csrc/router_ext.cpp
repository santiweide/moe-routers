#include <torch/extension.h>

#include <stdexcept>
#include <vector>

std::vector<torch::Tensor> topk_router_forward_cuda(torch::Tensor logits, int64_t k);

std::vector<torch::Tensor> topk_router_forward(torch::Tensor logits, int64_t k) {
  if (!logits.is_cuda()) {
    throw std::runtime_error("topk_router_forward: logits must be a CUDA tensor");
  }
  if (logits.dim() != 2) {
    throw std::runtime_error("topk_router_forward: logits must be rank-2 [tokens, experts]");
  }
  if (k <= 0) {
    throw std::runtime_error("topk_router_forward: k must be > 0");
  }
  // Kernel implementation is optimized for small k; keep it bounded.
  if (k > 8) {
    throw std::runtime_error("topk_router_forward: k must be <= 8 (kernel limitation)");
  }
  return topk_router_forward_cuda(logits, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &topk_router_forward, "MoE router top-k forward (CUDA)");
}


