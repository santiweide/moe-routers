import torch
import torch.nn.functional as F
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
import time

# 引入你的扩展
import fused_select_cuda as fsel
import grouped_ffn_cuda as ggemm

# ==========================================
# 1. 配置与初始化
# ==========================================
torch.manual_seed(42)
torch.set_grad_enabled(False) # 推理模式，省显存

T = 4096          # Token count
d_model = 4096    # Input dimension
E = 64            # Number of experts
hidden = 11008    # Hidden dimension (Llama-2-70B scale)

print(f"Config: T={T}, E={E}, d_model={d_model}, hidden={hidden}")
print("Initializing tensors...")

logits = torch.randn(T, E, device='cuda', dtype=torch.float16)
x = torch.randn(T, d_model, device='cuda', dtype=torch.float16)

# 权重 (Per-Expert)
W1 = torch.randn(E, d_model, hidden, device='cuda', dtype=torch.float16) * 0.02
b1 = torch.randn(E, hidden, device='cuda', dtype=torch.float16) * 0.01
W2 = torch.randn(E, hidden, d_model, device='cuda', dtype=torch.float16) * 0.02
b2 = torch.randn(E, d_model, device='cuda', dtype=torch.float16) * 0.01

# ==========================================
# 2. 定义 Forward 函数
# ==========================================
def run_moe_step():
    # --- 1. Router ---
    with record_function("1_Fused_Select_Router"):
        idx, w, counts, offsets, packed = fsel.fused_select_forward(logits, x)

    # --- 2. Data Split (CPU Overhead) ---
    # 这是一个同步点：CPU需要等待GPU算出counts，然后拷贝回CPU
    with record_function("2_Prepare_Inputs"):
        cpu_counts = counts.tolist() 
        split_inputs = list(torch.split(packed, cpu_counts, dim=0))
        
        active_indices = [i for i, c in enumerate(cpu_counts) if c > 0]
        active_inputs = [split_inputs[i] for i in active_indices]
        
        # Prepare weights (list slicing is fast on CPU)
        active_w1 = [W1[i] for i in active_indices]
        active_w2 = [W2[i] for i in active_indices]

    # --- 3. Layer 1 (GEMM + Bias + Act) ---
    with record_function("3_FFN_Layer1_GEMM"):
        # Grouped GEMM
        hidden_list = ggemm.grouped_ffn_forward(active_inputs, active_w1)

    with record_function("4_FFN_Layer1_BiasAct"):
        # Python loop overhead
        for i, expert_idx in enumerate(active_indices):
            hidden_list[i] = F.silu(hidden_list[i] + b1[expert_idx])

    # --- 4. Layer 2 (GEMM + Bias) ---
    with record_function("5_FFN_Layer2_GEMM"):
        output_list = ggemm.grouped_ffn_forward(hidden_list, active_w2)

    with record_function("6_FFN_Layer2_Bias"):
        for i, expert_idx in enumerate(active_indices):
            output_list[i] = output_list[i] + b2[expert_idx]

    # --- 5. Final Concat ---
    with record_function("7_Final_Reassemble"):
        final_out = torch.cat(output_list, dim=0)
    
    return final_out

# ==========================================
# 3. 简单耗时测试 (CUDA Events)
# ==========================================
print("\n>>> Warming up (10 iterations)...")
for _ in range(10):
    run_moe_step()

print(">>> Measuring Latency (20 iterations)...")
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
for _ in range(20):
    run_moe_step()
end_event.record()
torch.cuda.synchronize()

elapsed_time_ms = start_event.elapsed_time(end_event) / 20
print(f"Average Latency: {elapsed_time_ms:.3f} ms / step")


# ==========================================
# 4. Torch Profiler (生成 Trace)
# ==========================================
print("\n>>> Running Torch Profiler...")

# schedule: wait=1 (不记), warmup=1 (热身), active=1 (记录1次)
# repeat=1: 重复上述周期1次
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/moe_trace'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i in range(3):
        out = run_moe_step()
        prof.step() # 标记一步结束

print(f"Profiling done. Results saved to: ./log/moe_trace")
print("View results by running: pip install tensorboard && tensorboard --logdir=./log/moe_trace")
print("Or open the .json file in chrome://tracing")

# 打印简要统计表
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))