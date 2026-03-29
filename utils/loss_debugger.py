"""
Loss V4 Debug Tool: 用于分析和可视化 SimOTA 匹配情况

功能：
1. 统计正负样本比例
2. 分析小目标匹配情况
3. 可视化匹配结果
4. 诊断训练问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class LossV4Debugger:
    """DetectionBoxLossV4 调试工具"""
    
    def __init__(self, save_dir: str = "./debug_output"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.matching_history = []
        self.loss_history = []
    
    def log_matching(
        self,
        epoch: int,
        batch_idx: int,
        pos_mask: torch.Tensor,
        matched_ious: torch.Tensor,
        gt_boxes: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_areas: Optional[torch.Tensor] = None
    ):
        """记录匹配信息"""
        num_pos = pos_mask.sum().item()
        num_gt = gt_boxes.size(0)
        num_pred = pred_boxes.size(0)
        
        if gt_areas is None:
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        
        is_small = gt_areas < 0.01
        num_small_gt = is_small.sum().item()
        
        stats = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "num_pos": num_pos,
            "num_gt": num_gt,
            "num_pred": num_pred,
            "pos_ratio": num_pos / max(num_pred, 1),
            "pos_per_gt": num_pos / max(num_gt, 1),
            "num_small_gt": num_small_gt,
            "avg_iou": matched_ious.mean().item() if matched_ious.numel() > 0 else 0,
            "min_iou": matched_ious.min().item() if matched_ious.numel() > 0 else 0,
            "max_iou": matched_ious.max().item() if matched_ious.numel() > 0 else 0,
        }
        
        self.matching_history.append(stats)
        return stats
    
    def log_loss(
        self,
        epoch: int,
        loss_cls: float,
        loss_box: float,
        loss_iou: float,
        total_loss: float
    ):
        """记录损失信息"""
        self.loss_history.append({
            "epoch": epoch,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_iou": loss_iou,
            "total_loss": total_loss
        })
    
    def get_summary(self, last_n: int = 100) -> Dict:
        """获取最近N次匹配的统计摘要"""
        if len(self.matching_history) == 0:
            return {}
        
        recent = self.matching_history[-last_n:]
        
        return {
            "avg_pos_per_gt": np.mean([s["pos_per_gt"] for s in recent]),
            "avg_pos_ratio": np.mean([s["pos_ratio"] for s in recent]),
            "avg_iou": np.mean([s["avg_iou"] for s in recent]),
            "avg_min_iou": np.mean([s["min_iou"] for s in recent]),
            "zero_match_rate": sum(1 for s in recent if s["num_pos"] == 0) / len(recent),
            "total_samples": len(recent)
        }
    
    def plot_matching_trend(self, save_path: Optional[str] = None):
        """绘制匹配趋势图"""
        if len(self.matching_history) == 0:
            print("No matching history to plot")
            return
        
        epochs = [s["epoch"] for s in self.matching_history]
        pos_per_gt = [s["pos_per_gt"] for s in self.matching_history]
        avg_ious = [s["avg_iou"] for s in self.matching_history]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].scatter(epochs, pos_per_gt, alpha=0.5, s=10)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Positive Samples per GT")
        axes[0].set_title("Matching Trend: Positives per GT")
        axes[0].axhline(y=10, color='r', linestyle='--', label='Target topk=10')
        axes[0].legend()
        
        axes[1].scatter(epochs, avg_ious, alpha=0.5, s=10, color='green')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Average Matched IoU")
        axes[1].set_title("Matching Trend: Average IoU")
        axes[1].axhline(y=0.5, color='r', linestyle='--', label='Good IoU threshold')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.savefig(self.save_dir / "matching_trend.png", dpi=150)
        
        plt.close()
    
    def visualize_single_match(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        pos_mask: torch.Tensor,
        matched_gt_idx: torch.Tensor,
        matched_ious: torch.Tensor,
        img_size: int = 320,
        save_path: Optional[str] = None
    ):
        """可视化单张图片的匹配情况"""
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        for i, gt in enumerate(gt_boxes.cpu().numpy()):
            x1, y1, x2, y2 = gt * img_size
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='green', facecolor='none', label=f'GT {i}'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'GT{i}', color='green', fontsize=10, weight='bold')
        
        neg_mask = ~pos_mask
        neg_boxes = pred_boxes[neg_mask].cpu().numpy()
        if len(neg_boxes) > 0:
            sample_neg = neg_boxes[:min(50, len(neg_boxes))]
            for box in sample_neg:
                x1, y1, x2, y2 = box * img_size
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=0.5, edgecolor='gray', facecolor='none', alpha=0.3
                )
                ax.add_patch(rect)
        
        pos_boxes = pred_boxes[pos_mask].cpu().numpy()
        for j, (box, gt_idx, iou) in enumerate(zip(
            pos_boxes, matched_gt_idx.cpu().numpy(), matched_ious.cpu().numpy()
        )):
            x1, y1, x2, y2 = box * img_size
            color = plt.cm.RdYlGn(iou)
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y2 + 5, f'P{iou:.2f}', color=color, fontsize=8)
        
        ax.set_title(f"Matching Visualization\n"
                    f"GT: {len(gt_boxes)}, Pos: {pos_mask.sum()}, "
                    f"Avg IoU: {matched_ious.mean():.3f}")
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
        plt.colorbar(sm, ax=ax, label='IoU Score')
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        else:
            plt.savefig(self.save_dir / "match_visualization.png", dpi=150)
        
        plt.close()
    
    def diagnose(self) -> Dict[str, str]:
        """诊断潜在问题"""
        issues = []
        summary = self.get_summary()
        
        if not summary:
            return {"status": "No data to diagnose"}
        
        if summary["avg_pos_per_gt"] < 5:
            issues.append(f"Low positive samples per GT ({summary['avg_pos_per_gt']:.1f}), "
                         "consider increasing topk")
        
        if summary["avg_iou"] < 0.3:
            issues.append(f"Low average matched IoU ({summary['avg_iou']:.2f}), "
                         "check prediction quality or matching strategy")
        
        if summary["zero_match_rate"] > 0.1:
            issues.append(f"High zero-match rate ({summary['zero_match_rate']*100:.1f}%), "
                         "some GTs are not being matched at all")
        
        if len(self.loss_history) > 10:
            recent_losses = self.loss_history[-10:]
            cls_losses = [l["loss_cls"] for l in recent_losses]
            box_losses = [l["loss_box"] for l in recent_losses]
            
            if np.mean(cls_losses) > 5 * np.mean(box_losses):
                issues.append(f"Classification loss dominates, consider adjusting loss weights")
        
        return {
            "status": "OK" if len(issues) == 0 else "Issues found",
            "issues": issues if issues else ["No significant issues detected"],
            "summary": summary
        }


def run_debug_test():
    """运行调试测试"""
    debugger = LossV4Debugger()
    
    for epoch in range(5):
        for batch in range(10):
            num_pred = 1000
            num_gt = 5
            
            pred_boxes = torch.rand(num_pred, 4)
            pred_boxes[:, 2] = pred_boxes[:, 0] + torch.rand(num_pred) * 0.1
            pred_boxes[:, 3] = pred_boxes[:, 1] + torch.rand(num_pred) * 0.1
            
            gt_boxes = torch.rand(num_gt, 4)
            gt_boxes[:, 2] = gt_boxes[:, 0] + torch.rand(num_gt) * 0.05 + 0.02
            gt_boxes[:, 3] = gt_boxes[:, 1] + torch.rand(num_gt) * 0.05 + 0.02
            
            pos_mask = torch.rand(num_pred) > 0.95
            matched_ious = torch.rand(pos_mask.sum()) * 0.5 + 0.3
            
            debugger.log_matching(epoch, batch, pos_mask, matched_ious, gt_boxes, pred_boxes)
    
    print("Summary:", debugger.get_summary())
    print("\nDiagnosis:", debugger.diagnose())
    debugger.plot_matching_trend()


if __name__ == "__main__":
    run_debug_test()