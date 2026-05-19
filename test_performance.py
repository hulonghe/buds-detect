import torch
import os
import glob
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops_with_torch_profiler
from nn.bbox.EnhancedDynamicBoxDetector_ablation import DynamicBoxDetector

# ============================================================
# ⚙️ 配置
# ============================================================
IMG_SIZE = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_ITERS = 50
TEST_ITERS = 200

# 模型路径
CUSTOM_MODEL_PATHS = [
    "./runs/daC-m_default-ep100-si320-lr0_001-wa1-baresnet34-iociou-bososa-clfocal-io0_15-sc0_4-ga1_0-al0_5-dr0_01-fpTrue-trTrue-reTrue/model_epoch_best.pth",
]
YOLO_MODEL_PATHS = glob.glob("other_models/*.pt")


# ============================================================
# 🧩 导入自定义模型
# ============================================================
def load_custom_model(weight_path, device):
    model_kwargs = dict(
        in_dim=IMG_SIZE, hidden_dim=128, nhead=4, num_layers=2, cls_num=1,
        once_embed=True, is_split_trans=False, is_fpn=True,
        dropout=0.0, backbone_type="resnet34", device=device,
        use_transformer=True, use_refine=True
    )
    model = DynamicBoxDetector(**model_kwargs)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


# ============================================================
# 🔍 Benchmark 通用函数
# ============================================================
def benchmark_model(model, model_name, weight_path, imgsz=320):
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(DEVICE)

    # 1️⃣ 参数量
    params = sum(p.numel() for p in model.parameters())

    # 2️⃣ FLOPs（统一使用 Ultralytics 的 torch_profiler）
    with torch.no_grad():
        flops = get_flops_with_torch_profiler(model, imgsz=imgsz)

    # 3️⃣ 模型文件大小
    size_mb = os.path.getsize(weight_path) / 1e6

    # 4️⃣ 推理速度
    torch.cuda.empty_cache()
    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            _ = model(dummy_input)

    times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(TEST_ITERS):
        with torch.no_grad():
            start_event.record()
            _ = model(dummy_input)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))

    avg_time = sum(times) / len(times)
    fps = 1000.0 / avg_time

    # 5️⃣ 显存占用
    mem_mb = torch.cuda.memory_allocated() / 1024 ** 2 if DEVICE == "cuda" else 0

    print(f"\n===> Testing model: {model_name}")
    print(f" - Params: {params}")
    print(f" - GFLOPs:  {flops}")
    print(f" - Size:   {size_mb:.2f} MB")
    print(f" - FPS:    {fps:.2f}")
    print(f" - Mem:    {mem_mb:.2f} MB")

    return dict(
        Model=model_name,
        Params=params,
        GFLOPs=round(flops, 6),
        FPS=round(fps, 2),
        Size_MB=round(size_mb, 2),
        Mem_MB=round(mem_mb, 2),
    )


# ============================================================
# 🚀 主流程
# ============================================================
results = []

# 🤖 测试 YOLO 模型
for path in YOLO_MODEL_PATHS:
    name = os.path.basename(path)
    yolo = YOLO(path)
    model = yolo.model.to(DEVICE).eval()
    results.append(benchmark_model(model, name, path, imgsz=IMG_SIZE))

# 🧠 测试自定义模型
for path in CUSTOM_MODEL_PATHS:
    model = load_custom_model(path, DEVICE)
    name = os.path.basename(path)
    results.append(benchmark_model(model, name, path, imgsz=IMG_SIZE))

# ============================================================
# 📊 汇总结果
# ============================================================
df = pd.DataFrame(results, columns=["Model", "Params", "GFLOPs", "FPS", "Size_MB", "Mem_MB"])
print("\n=================== Benchmark Summary ===================")
print(df.to_markdown(index=False))
print("==========================================================")

# 保存结果
os.makedirs("save_logs", exist_ok=True)
df.to_csv("save_logs/all_benchmark_results.csv", index=False)
print("\n✅ Results saved to save_logs/all_benchmark_results.csv")
