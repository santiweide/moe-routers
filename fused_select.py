import torch
import torch.nn.functional as F
import fused_select_cuda as fsel
import grouped_ffn_cuda as ggemm

# ==========================================
# 1. Configuration & Initialization
# ==========================================
torch.manual_seed(42)

T = 4096          # Total tokens
d_model = 4096    # Input dimension
E = 64            # Number of experts
hidden = 11008    # Hidden dimension (e.g., Llama-2-70B SwiGLU size)

print(f"Config: T={T}, E={E}, d_model={d_model}, hidden={hidden}")

# Initialize Inputs (FP16)
logits = torch.randn(T, E, device='cuda', dtype=torch.float16)
x = torch.randn(T, d_model, device='cuda', dtype=torch.float16)

# ==========================================
# 2. Weight Initialization (Fixing your NameError)
# ==========================================
# Note on Shapes for Grouped GEMM (C++ Kernel is RowMajor):
# Matrix Mult: Input[M, K] @ Weight[K, N] = Output[M, N]

# Layer 1: d_model -> hidden
# W1 Shape: [E, d_model, hidden] 
W1 = torch.randn(E, d_model, hidden, device='cuda', dtype=torch.float16) * 0.02
b1 = torch.randn(E, hidden, device='cuda', dtype=torch.float16) * 0.01

# Layer 2: hidden -> d_model
# W2 Shape: [E, hidden, d_model]
W2 = torch.randn(E, hidden, d_model, device='cuda', dtype=torch.float16) * 0.02
b2 = torch.randn(E, d_model, device='cuda', dtype=torch.float16) * 0.01

# ==========================================
# 3. Router (Fused Select)
# ==========================================
print(">>> Running Fused Select (Router)...")
# idx: [T, 2], counts: [E], packed: [Sum(counts), d_model]
idx, w, counts, offsets, packed = fsel.fused_select_forward(logits, x)

print(f"Packed shape: {packed.shape}")

# ==========================================
# 4. Data Preparation (Split for Grouped GEMM)
# ==========================================
# Convert counts to CPU list for torch.split
cpu_counts = counts.tolist()

# Split the huge packed tensor into a list of smaller tensors (one per expert)
split_inputs = list(torch.split(packed, cpu_counts, dim=0))

# Filter out empty experts (Grouped GEMM cannot handle size 0)
active_indices = [i for i, c in enumerate(cpu_counts) if c > 0]
active_inputs = [split_inputs[i] for i in active_indices]

print(f"Active experts: {len(active_indices)} / {E}")

# ==========================================
# 5. FFN Layer 1 (GEMM + Bias + Activation)
# ==========================================
print(">>> Running FFN Layer 1...")

# Prepare W1 weights for active experts
active_w1 = [W1[i] for i in active_indices]

# 1. Run GEMM: x @ W1
hidden_list = ggemm.grouped_ffn_forward(active_inputs, active_w1)

# 2. Run Bias + SiLU (Python Loop)
for i, expert_idx in enumerate(active_indices):
    # Add bias
    hidden_list[i] = hidden_list[i] + b1[expert_idx]
    # Activation
    hidden_list[i] = F.silu(hidden_list[i])

# ==========================================
# 6. FFN Layer 2 (GEMM + Bias)
# ==========================================
print(">>> Running FFN Layer 2...")

# Prepare W2 weights for active experts
active_w2 = [W2[i] for i in active_indices]

# 3. Run GEMM: hidden @ W2
output_list = ggemm.grouped_ffn_forward(hidden_list, active_w2)

# 4. Run Bias (Python Loop)
for i, expert_idx in enumerate(active_indices):
    output_list[i] = output_list[i] + b2[expert_idx]

# ==========================================
# 7. Reassemble (Optional)
# ==========================================
# Concatenate back to packed shape
final_output = torch.cat(output_list, dim=0)

print("Done.")
print(f"Final Output Shape: {final_output.shape}") 
# Expected: [T*2, d_model] (since top-k=2)