#include <torch/extension.h>

#include <stdexcept>
#include <vector>

std::vector<torch::Tensor> topk_router_forward_cuda(torch::Tensor logits, int64_t k);

std::vector<torch::Tensor> topk_router_forward(torch::Tensor logits, int64_t k) {
  TORCH_CHECK(logits.is_cuda(), "topk_router_forward: logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 2, "topk_router_forward: logits must be rank-2 [tokens, experts]");
  TORCH_CHECK(k > 0, "topk_router_forward: k must be > 0");
  // Kernel implementation is optimized for small k; keep it bounded.
  TORCH_CHECK(k <= 8, "topk_router_forward: k must be <= 8 (kernel limitation)");
  TORCH_CHECK(k <= logits.size(1), "topk_router_forward: k must be <= experts (logits.size(1))");

  // The CUDA kernel expects row-major contiguous memory for best performance.
  TORCH_CHECK(
      logits.is_contiguous(),
      "topk_router_forward: logits must be contiguous; call logits = logits.contiguous()");

  return topk_router_forward_cuda(logits, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &topk_router_forward, "MoE router top-k forward (CUDA)");
}


