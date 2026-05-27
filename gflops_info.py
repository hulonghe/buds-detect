from thop import profile, clever_format
from nn.bbox.EnhancedDynamicBoxDetector_ablation import DynamicBoxDetector
import torch
import time
import psutil
import os

model_kwargs = dict(
    in_dim=320, hidden_dim=128, nhead=4, num_layers=2, cls_num=1,
    once_embed=True, is_split_trans=False, is_fpn=True,
    dropout=0.01, backbone_type="resnet18", device="cpu",
    use_transformer=True, use_refine=True
)

model = DynamicBoxDetector(**model_kwargs)

# 1. 切换到评估模式
model.eval()

# 2. 准备输入张量
input_tensor = torch.randn(1, 3, 320, 320).to(model.device)

# 3. 统计 MACs 和参数量
macs, params = profile(model, inputs=(input_tensor,), verbose=False)
formatted_macs, formatted_params = clever_format([macs * 2, params], "%.3f")
print(f"GFLOPs: {formatted_macs}, Params: {int(params):,} ({formatted_params})")

# ================= FPS 与 资源消耗 计算部分 =================
# 获取当前进程，用于监控 CPU 和内存
process = psutil.Process(os.getpid())

# 4. 预热运行 (Warm-up)
warmup_iters = 50
for _ in range(warmup_iters):
    with torch.no_grad():
        _ = model(input_tensor)

# 5. 正式计时与资源监控
test_iters = 200
start_time = time.time()
with torch.no_grad():
    for _ in range(test_iters):
        _ = model(input_tensor)

# 如果是 CUDA 设备，强制同步确保 GPU 运算全部完成
if torch.cuda.is_available():
    torch.cuda.synchronize()
    
end_time = time.time()

# 6. 计算 FPS
total_time = end_time - start_time
fps = test_iters / total_time
avg_inference_time_ms = (total_time / test_iters) * 1000

# 7. 获取资源消耗数据
cpu_percent = process.cpu_percent(interval=1) # 获取最近1秒的CPU占用率
memory_info = process.memory_info()
system_memory_mb = memory_info.rss / 1024 / 1024  # 系统内存占用 (MB)

# 打印 CPU 和 系统内存信息
print(f"FPS: {fps:.2f}, Avg Time: {avg_inference_time_ms:.2f} ms")
print(f"CPU Usage: {cpu_percent:.1f}%, System Memory: {system_memory_mb:.2f} MB")

# 8. 获取显存消耗 (如果使用了 GPU)
if torch.cuda.is_available():
    # 获取当前进程的显存占用
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    print(f"GPU Memory Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
else:
    print("当前未使用 GPU，无显存消耗数据。")