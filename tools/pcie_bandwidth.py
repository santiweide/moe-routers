import torch
import time

def measure_p2p_bandwidth(device_src, device_dst, size_mb=1024, num_repeats=10):
    """
    测量两个 GPU 之间的 P2P 拷贝带宽
    size_mb: 每次传输的数据大小 (MB)
    num_repeats: 重复测试次数取平均
    """
    if torch.cuda.device_count() < 2:
        print("错误：需要至少 2 个 GPU 才能运行此测试。")
        return

    print(f"--- 开始测试: GPU {device_src} -> GPU {device_dst} ---")
    
    # 1. 检查 P2P 访问权限 (Peer Access)
    can_access = torch.cuda.can_device_access_peer(device_src, device_dst)
    print(f"P2P Access (Peer-to-Peer) 状态: {'开启' if can_access else '关闭 (需要经过 CPU)'}")
    if not can_access:
        print("注意: 由于 P2P 关闭，数据将通过 System Memory 转发，速度会受限于 QPI/UPI 总线。")

    # 2. 准备数据 (Float32占用4字节，所以元素数量 = MB * 1024 * 1024 / 4)
    num_elements = int(size_mb * 1024 * 1024 / 4)
    
    try:
        # 在源设备创建张量
        src_tensor = torch.randn(num_elements, device=f'cuda:{device_src}')
        # 在目标设备预分配空间
        dst_tensor = torch.empty(num_elements, device=f'cuda:{device_dst}')
    except Exception as e:
        print(f"显存不足，无法分配 {size_mb}MB 数据。请尝试减小 size_mb。")
        return

    # 3. 预热 (Warm-up)
    # 让 GPU 频率提升，缓存填充，避免首次启动延迟影响
    for _ in range(5):
        dst_tensor.copy_(src_tensor)
    torch.cuda.synchronize()

    # 4. 正式测速
    # 使用 CUDA Event 进行纳秒级精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_repeats):
        # 核心传输操作
        dst_tensor.copy_(src_tensor)
    end_event.record()

    # 等待 GPU 完成所有操作
    end_event.synchronize()

    # 5. 计算带宽
    elapsed_time_ms = start_event.elapsed_time(end_event) # 毫秒
    total_bytes = (size_mb * 1024 * 1024) * num_repeats
    bandwidth_gbs = (total_bytes / (elapsed_time_ms / 1000)) / (1024**3)

    print(f"传输大小: {size_mb} MB")
    print(f"平均耗时: {elapsed_time_ms / num_repeats:.2f} ms")
    print(f"实测带宽: {bandwidth_gbs:.2f} GB/s")
    print("-" * 40 + "\n")

if __name__ == "__main__":
    # 检查是否有 GPU
    if not torch.cuda.is_available():
        print("没有检测到 CUDA 环境")
    else:
        # 测试 GPU 0 -> GPU 1
        measure_p2p_bandwidth(0, 1, size_mb=512) # 测试 512MB 数据块
